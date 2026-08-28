use polars::error::PolarsError;
use polars::frame::DataFrame;
use polars::prelude::*;
use polars::series::IntoSeries;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Mutex;

// Internal modules
mod consolidation;
mod lgbm_parser;

// Re-export public functions from lgbm_parser
pub use lgbm_parser::{
    build_analysis_tablemodel, build_consolidated_tablemodel, process_lgbm_trees,
};

// Re-export public functions from consolidation
pub use consolidation::{combine_all_tables, expand_and_combine_tables};

// Begin Metadata Structures

/// How many free parameters a table's rows represent.
///
/// This does not change how a table is *read* — lookup is a step lookup either way,
/// and a deployed rating engine cannot tell the difference. It changes how many
/// degrees of freedom the fit spends on the table.
#[derive(Debug, Clone, PartialEq)]
pub enum TableSemantics {
    /// Every row carries its own free factor. A five-row table spends four parameters
    /// once the model is anchored. This is what a LightGBM-converted table is, and the
    /// default.
    Step,
    /// The row factors are constrained to a polynomial in a value attached to each
    /// row: `factor[r] = sum over m of beta_m * values[r]^m`, up to the constant the
    /// intercept absorbs. A table spends `degree` parameters, whatever its row count —
    /// one for a straight line, two for a curve that bends once.
    ///
    /// This is the classical actuarial *variate*: age entered as a continuous driver
    /// rather than as a set of independent levels. Three things follow from it that
    /// free levels do not give you:
    ///
    /// * The fitted curve is smooth by construction, not by penalty, and at degree 1
    ///   monotone as well.
    /// * Rows with little or no exposure still get a sensible factor, read off the
    ///   curve rather than left at their starting value.
    /// * The table is still an ordinary step table, so it deploys unchanged.
    ///
    /// `values` is one number per row: what that row is worth on the driver's scale.
    /// It is supplied rather than derived from the table's own numeric column, because
    /// that column holds inclusive bin *upper bounds* — the top bin's bound is normally
    /// infinite, and a bound is the edge of a bin rather than a point inside it. See
    /// [`RatingTable::as_variate`] and [`RatingTable::as_polynomial_variate`].
    Variate { values: Vec<f64>, degree: usize },
}

/// Centre and half-range of a variate's values, mapping them onto `[-1, 1]`.
///
/// Powers of a raw driver are a bad basis: age to the fourth is around ten million
/// while age itself is around forty, so the normal matrix spans orders of magnitude and
/// the solve loses most of its precision. Mapping onto `[-1, 1]` first keeps every
/// power in the same range. It changes nothing about the fit — the span of
/// `{1, u, u^2, ...}` is the span of `{1, v, v^2, ...}` — only its conditioning.
///
/// `None` when every value is identical, which leaves no range to scale by.
pub fn variate_basis_params(values: &[f64]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in values {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    if !(hi > lo) {
        return None;
    }
    Some(((lo + hi) / 2.0, (hi - lo) / 2.0))
}

/// Binomial coefficient, exact for the small degrees a variate allows.
fn binomial(n: usize, k: usize) -> f64 {
    let mut result = 1.0f64;
    for i in 0..k {
        result = result * (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// The most a variate's degree may be, regardless of how many rows the table has.
///
/// Well past anything defensible: high-degree polynomials oscillate between the points
/// they interpolate, which is the opposite of what a variate is for. If a curve needs
/// more shape than this, it needs a different basis, not a higher power.
pub const MAX_VARIATE_DEGREE: usize = 8;

impl Default for TableSemantics {
    fn default() -> Self {
        TableSemantics::Step
    }
}

/// Metadata for a RatingTable
#[derive(Debug, Clone)]
pub struct TableMetadata {
    pub name: String,
    pub is_offset: bool,    // Table is fixed, not updated by GLM
    pub is_updatable: bool, // Can GLM update this table's factors?
    /// How many free parameters this table's rows represent. See [`TableSemantics`].
    pub semantics: TableSemantics,
}

impl Default for TableMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            is_offset: false,
            is_updatable: true,
            semantics: TableSemantics::default(),
        }
    }
}

/// Metadata for individual rows within a RatingTable
#[derive(Debug, Clone)]
pub struct RowMetadata {
    pub is_offset: bool, // Row is locked, not updated by GLM
}

impl Default for RowMetadata {
    fn default() -> Self {
        Self { is_offset: false }
    }
}

// Begin RatingTable implementation
#[derive(Debug, Clone)]
enum FeatureType {
    Numeric,
    Categorical,
}

#[derive(Debug, Clone)]
pub enum LinkFunction {
    Identity, // for 'regression'
    Logit,    // for 'binary'
    Log,      // for 'poisson', 'gamma', 'tweedie'
}

impl LinkFunction {
    pub fn from_objective(objective: &str) -> Self {
        match objective.to_lowercase().as_str() {
            "regression" | "gaussian" => LinkFunction::Identity,
            "binary" | "binomial" | "logistic" => LinkFunction::Logit,
            "poisson" => LinkFunction::Log,
            "tweedie" => LinkFunction::Log,
            "gamma" => LinkFunction::Log,
            _ => LinkFunction::Identity, // default to identity for unknown objectives
        }
    }

    fn inverse(&self, x: f64) -> f64 {
        match self {
            LinkFunction::Identity => x,
            LinkFunction::Logit => 1.0 / (1.0 + (-x).exp()),
            LinkFunction::Log => x.exp(),
        }
    }

    fn to_string(&self) -> String {
        match self {
            LinkFunction::Identity => "identity".to_string(),
            LinkFunction::Logit => "logit".to_string(),
            LinkFunction::Log => "log".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FeatureValue {
    Numeric(f64),
    Categorical(i32),
}

impl From<f64> for FeatureValue {
    fn from(v: f64) -> Self {
        if v.fract() == 0.0 && v >= -999.0 {
            FeatureValue::Categorical(v as i32)
        } else {
            FeatureValue::Numeric(v)
        }
    }
}

impl From<i32> for FeatureValue {
    fn from(v: i32) -> Self {
        FeatureValue::Categorical(v)
    }
}

#[derive(Debug, Clone)]
pub struct RatingTable {
    pub data: DataFrame,
    // Cache column metadata
    numeric_columns: HashMap<String, usize>, // column name -> index
    categorical_columns: HashMap<String, usize>, // column name -> index
    // NEW: Metadata for table and row-level behavior
    pub metadata: TableMetadata,
    pub row_metadata: Option<Vec<RowMetadata>>,
}

impl RatingTable {
    pub fn new(data: DataFrame, _existing_row_number_col: Option<&str>) -> Self {
        // Remove row_number handling from constructor since we'll generate it on demand
        let mut numeric_columns = HashMap::new();
        let mut categorical_columns = HashMap::new();

        for (idx, col_name) in data.get_column_names().iter().enumerate() {
            if col_name == &"Rating_Factor" {
                continue;
            }
            match data.column(col_name).unwrap().dtype() {
                DataType::Float64 => {
                    numeric_columns.insert(col_name.to_string(), idx);
                }
                DataType::Int32 => {
                    categorical_columns.insert(col_name.to_string(), idx);
                }
                _ => continue,
            }
        }

        Self {
            data: data.clone(),
            numeric_columns,
            categorical_columns,
            metadata: TableMetadata::default(),
            row_metadata: None, // Lazy initialization - only create if needed
        }
    }

    pub fn find_row_match(&self, feature_values: &HashMap<String, FeatureValue>) -> Option<usize> {
        // Early return if we don't have all required features
        if !self.has_all_required_features(feature_values) {
            return None;
        }

        // ⭐ OPTIMIZATION: Pre-extract and cache input values + column references
        // CRITICAL: Iterate once to ensure consistent ordering between inputs and columns
        let mut categorical_inputs = Vec::with_capacity(self.categorical_columns.len());
        let categorical_columns_vec: Vec<_> = self
            .categorical_columns
            .iter()
            .map(|(col_name, &col_idx)| {
                // Build inputs vector in the same iteration to guarantee order consistency
                if let Some(FeatureValue::Categorical(input_cat)) = feature_values.get(col_name) {
                    categorical_inputs.push(Some(*input_cat));
                } else {
                    categorical_inputs.push(None);
                }
                // Return the cached column reference
                self.data.get_columns()[col_idx].i32().unwrap()
            })
            .collect();

        let mut numeric_inputs = Vec::with_capacity(self.numeric_columns.len());
        let numeric_columns_vec: Vec<_> = self
            .numeric_columns
            .iter()
            .map(|(col_name, &col_idx)| {
                // Build inputs vector in the same iteration to guarantee order consistency
                if let Some(FeatureValue::Numeric(input_val)) = feature_values.get(col_name) {
                    numeric_inputs.push(Some(*input_val));
                } else {
                    numeric_inputs.push(None);
                }
                // Return the cached column reference
                self.data.get_columns()[col_idx].f64().unwrap()
            })
            .collect();

        let mut best_row: Option<usize> = None;
        let mut used_wildcard = false;

        'row_loop: for i in 0..self.data.height() {
            let mut row_matches = true;
            let mut this_row_used_wildcard = false;

            // ⭐ OPTIMIZATION 3: Direct array access instead of HashMap lookups
            // Check categorical features first
            for (idx, input_cat_opt) in categorical_inputs.iter().enumerate() {
                if let Some(input_cat) = input_cat_opt {
                    let col_val = unsafe { categorical_columns_vec[idx].get_unchecked(i) };

                    if let Some(table_cat) = col_val {
                        if table_cat != -999 && table_cat != *input_cat {
                            row_matches = false;
                            break;
                        }
                        if table_cat == -999 {
                            this_row_used_wildcard = true;
                        }
                    }
                }
            }

            if !row_matches {
                continue;
            }

            // Check numeric features with cached column references
            for (idx, input_val_opt) in numeric_inputs.iter().enumerate() {
                if let Some(input_val) = input_val_opt {
                    let col_val = unsafe { numeric_columns_vec[idx].get_unchecked(i) };

                    if let Some(threshold) = col_val {
                        if input_val > &threshold {
                            continue 'row_loop;
                        }
                    }
                }
            }

            // If we get here, we found a match
            // Only update if this is the first match or if we found a more specific match
            if best_row.is_none() || (used_wildcard && !this_row_used_wildcard) {
                best_row = Some(i);
                used_wildcard = this_row_used_wildcard;
            }
        }

        best_row
    }

    // Public accessors for vectorized matching
    pub fn get_categorical_columns(&self) -> &HashMap<String, usize> {
        &self.categorical_columns
    }

    pub fn get_numeric_columns(&self) -> &HashMap<String, usize> {
        &self.numeric_columns
    }

    pub fn get_rating_factor(&self, row: usize) -> f64 {
        let rating_col = self.data.column("Rating_Factor").unwrap();
        unsafe {
            rating_col
                .f64()
                .unwrap()
                .get_unchecked(row)
                .unwrap_or(f64::NAN)
        }
    }

    pub fn predict(&self, feature_values: &HashMap<String, FeatureValue>) -> f64 {
        // Matches a row to the feature values and returns the rating factor, does not apply the link function
        match self.find_row_match(feature_values) {
            Some(row) => {
                // Get rating factor from the matching row
                self.get_rating_factor(row)
            }
            None => f64::NAN,
        }
    }

    pub fn get_rating_factor_batch(&self, row_nums: Series) -> Result<Vec<f64>, PolarsError> {
        let mut df = self.data.clone();

        // Create temporary row numbers for joining
        let n_rows = df.height();
        let temp_row_numbers = Series::new(
            "row_number".into(),
            (0..n_rows).map(|x| x as i32).collect::<Vec<i32>>(),
        );
        let df_with_rownums = df.with_column(temp_row_numbers)?;

        let mut row_number = row_nums.clone();
        row_number.rename("row_number".into());
        let out_df = DataFrame::new(vec![row_number.into()])?;

        // join using temporary row numbers
        let joined = out_df.join(
            &df_with_rownums,
            ["row_number"],
            ["row_number"],
            JoinArgs::new(JoinType::Left),
            None,
        )?;
        Ok(joined
            .column("Rating_Factor")?
            .f64()?
            .into_iter()
            .flatten()
            .collect())
    }

    pub fn predict_batch(&self, df: &DataFrame) -> Vec<f64> {
        // Predicts a batch of rows, does not apply the link function
        // Uses parallel processing if the number of rows is greater than the ROW_PARALLEL_THRESHOLD
        // Otherwise, it processes rows sequentially
        const ROW_PARALLEL_THRESHOLD: usize = 10;
        let n_rows = df.height();
        if n_rows > ROW_PARALLEL_THRESHOLD {
            (0..n_rows)
                .into_par_iter()
                .map(|row_idx| match self.extract_row_features(df, row_idx) {
                    Ok(features) => self.predict(&features),
                    Err(_) => f64::NAN,
                })
                .collect()
        } else {
            (0..n_rows)
                .map(|row_idx| match self.extract_row_features(df, row_idx) {
                    Ok(features) => self.predict(&features),
                    Err(_) => f64::NAN,
                })
                .collect()
        }
    }

    fn round_rating_factor(&self, num_decimals: i32) -> RatingTable {
        // Create a copy of the DataFrame
        let mut new_df = self.data.clone();

        // Get the Rating_Factor column, round it, and replace it in the DataFrame
        let rounded = new_df
            .column("Rating_Factor")
            .unwrap()
            .f64()
            .unwrap()
            .apply(|x| match x {
                Some(val) => {
                    Some((val * 10f64.powi(num_decimals)).round() / 10f64.powi(num_decimals))
                }
                None => None,
            })
            .into_series();

        new_df.with_column(rounded).unwrap();

        // Create and return a new RatingTable with the rounded values
        RatingTable::new(new_df, None)
    }

    // Helper methods
    #[inline]
    fn has_all_required_features(&self, feature_values: &HashMap<String, FeatureValue>) -> bool {
        // Check that all features required by the table are present in the input
        self.numeric_columns
            .keys()
            .all(|k| feature_values.contains_key(k))
            && self
                .categorical_columns
                .keys()
                .all(|k| feature_values.contains_key(k))
    }

    // Add new method to get feature info
    pub fn get_feature_info(&self) -> HashMap<String, DataType> {
        self.data
            .get_column_names()
            .iter()
            .filter(|&name| *name != "Rating_Factor")
            .map(|name| {
                (
                    name.to_string(),
                    self.data.column(name).unwrap().dtype().clone(),
                )
            })
            .collect()
    }

    // Add extract_row_features method to RatingTable
    fn extract_row_features(
        &self,
        df: &DataFrame,
        row_idx: usize,
    ) -> Result<HashMap<String, FeatureValue>, PolarsError> {
        let mut feature_values = HashMap::new();

        for col_name in df.get_column_names() {
            let column = df.column(col_name)?;
            match column.dtype() {
                DataType::Float64 => {
                    if let Some(value) = column.f64()?.get(row_idx) {
                        feature_values.insert(col_name.to_string(), FeatureValue::Numeric(value));
                    }
                }
                DataType::Int32 => {
                    if let Some(value) = column.i32()?.get(row_idx) {
                        feature_values
                            .insert(col_name.to_string(), FeatureValue::Categorical(value));
                    }
                }
                _ => continue,
            }
        }

        Ok(feature_values)
    }

    pub fn one_way_analysis_table(
        &self,
        df: &DataFrame, // ⭐ REFERENCE
        target_column: &str,
        weight_column: Option<&str>,
    ) -> Result<DataFrame, PolarsError> {
        crate::analysis::one_way_analysis_table(self, df, target_column, weight_column)
        // ⭐ NO CLONE
    }

    // NEW: Offset-related methods

    /// Mark this entire table as an offset (fixed, not updated by GLM)
    pub fn as_offset(mut self) -> Self {
        self.metadata.is_offset = true;
        self.metadata.is_updatable = false;
        self
    }

    /// Set a specific row as offset (locked, not updated by GLM)
    pub fn set_row_offset(&mut self, row_idx: usize, is_offset: bool) {
        // Lazy initialization of row metadata
        if self.row_metadata.is_none() {
            self.row_metadata = Some(vec![RowMetadata::default(); self.data.height()]);
        }

        if row_idx < self.data.height() {
            self.row_metadata.as_mut().unwrap()[row_idx].is_offset = is_offset;
        }
    }

    /// Check if a specific row is marked as offset
    pub fn is_row_offset(&self, row_idx: usize) -> bool {
        self.row_metadata
            .as_ref()
            .and_then(|meta| meta.get(row_idx))
            .map(|m| m.is_offset)
            .unwrap_or(false)
    }

    // Variate methods

    /// Constrains this table's factors to a straight line through `values`, so the
    /// whole table costs one parameter instead of one per row.
    ///
    /// `values` gives what each row is worth on the driver's scale, in row order. For
    /// an age table with bounds `[20, 30, 40, 50, inf]` a natural choice is
    /// `[20, 30, 40, 50, 65]` — note the last entry stands in for the open-ended top
    /// bin, which is precisely why these are supplied rather than taken from the
    /// table's own column.
    ///
    /// Lookup is unaffected: the table is still read by its bounds, and the fitted
    /// factors still sit in `Rating_Factor`. The only difference is that all of them
    /// will lie exactly on a line.
    ///
    /// ```text
    ///  Age (bound)   values    Rating_Factor after fitting
    ///        20        20            0.0000        <- anchored base
    ///        30        30            0.0850
    ///        40        40            0.1700
    ///        50        50            0.2550
    ///       inf        65            0.3825
    /// ```
    ///
    /// For a curve rather than a line, see [`RatingTable::as_polynomial_variate`].
    pub fn as_variate(self, values: Vec<f64>) -> Result<Self, PolarsError> {
        self.as_polynomial_variate(values, 1)
    }

    /// As [`RatingTable::as_variate`], but fits a polynomial of the given degree
    /// instead of a straight line: `factor[r] = sum of beta_m * values[r]^m`.
    ///
    /// Degree 1 is a line and costs one parameter; degree 2 bends once and costs two;
    /// and so on. The table always keeps every row and is always read as a step table
    /// — the degree only decides how many parameters the fit spends describing the
    /// shape.
    ///
    /// The degree cannot reach the number of distinct values: at `distinct - 1` the
    /// polynomial already passes exactly through every row, which is the same fit as
    /// free levels, and beyond that the extra terms are not identified.
    pub fn as_polynomial_variate(
        mut self,
        values: Vec<f64>,
        degree: usize,
    ) -> Result<Self, PolarsError> {
        let label = if self.metadata.name.is_empty() {
            "table".to_string()
        } else {
            format!("table '{}'", self.metadata.name)
        };
        let n_rows = self.data.height();

        if values.len() != n_rows {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot make {} a variate: got {} values for {} rows. Supply one value \
                     per row, in row order.",
                    label,
                    values.len(),
                    n_rows
                )
                .into(),
            ));
        }

        for (i, v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Cannot make {} a variate: value {} for row {} is not finite. These \
                         are points on the driver's scale, not bin bounds - for an \
                         open-ended top bin, choose a representative value such as the \
                         exposure-weighted mean.",
                        label, v, i
                    )
                    .into(),
                ));
            }
        }

        if degree == 0 {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot make {} a variate of degree 0: that is a constant, which the \
                     intercept already carries.",
                    label
                )
                .into(),
            ));
        }
        if degree > MAX_VARIATE_DEGREE {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot make {} a variate of degree {}: the limit is {}. High-degree \
                     polynomials oscillate between the points they pass through, which is \
                     the opposite of what a variate is for.",
                    label, degree, MAX_VARIATE_DEGREE
                )
                .into(),
            ));
        }

        let mut distinct: Vec<f64> = values.clone();
        distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distinct.dedup();
        if distinct.len() == 1 {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot make {} a variate: all {} values are {}, so there is no variation \
                     to fit a curve through and no slope to estimate.",
                    label, n_rows, values[0]
                )
                .into(),
            ));
        }
        if distinct.len() <= degree {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot make {} a variate of degree {}: its values take only {} distinct \
                     value(s), so a degree-{} polynomial already passes through every row and \
                     the higher terms are not identified. Use degree {} or lower.",
                    label,
                    degree,
                    distinct.len(),
                    distinct.len() - 1,
                    distinct.len() - 1
                )
                .into(),
            ));
        }

        if (0..n_rows).any(|r| self.is_row_offset(r)) {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot make {} a variate: it has locked rows. Every factor is derived \
                     from the fitted curve, so pinning a single row would break it. Lock the \
                     whole table with as_offset() instead.",
                    label
                )
                .into(),
            ));
        }

        self.metadata.semantics = TableSemantics::Variate { values, degree };
        Ok(self)
    }

    /// How many free parameters this table's rows represent.
    pub fn semantics(&self) -> &TableSemantics {
        &self.metadata.semantics
    }

    /// The per-row values behind a variate table, or `None` for a step table.
    pub fn variate_values(&self) -> Option<&[f64]> {
        match &self.metadata.semantics {
            TableSemantics::Variate { values, .. } => Some(values),
            TableSemantics::Step => None,
        }
    }

    /// The polynomial degree of a variate table, or `None` for a step table.
    pub fn variate_degree(&self) -> Option<usize> {
        match &self.metadata.semantics {
            TableSemantics::Variate { degree, .. } => Some(*degree),
            TableSemantics::Step => None,
        }
    }

    /// The fitted slope of a *linear* variate table, recovered from any two rows whose
    /// values differ.
    ///
    /// `None` for a step table, and for a variate of degree above 1 — a curve has no
    /// single slope. Use [`RatingTable::variate_coefficients`] there.
    pub fn variate_slope(&self) -> Option<f64> {
        if self.variate_degree()? != 1 {
            return None;
        }
        let values = self.variate_values()?;
        let v0 = values[0];
        let r = values.iter().position(|v| *v != v0)?;
        Some((self.get_rating_factor(r) - self.get_rating_factor(0)) / (values[r] - v0))
    }

    /// The fitted polynomial coefficients `[beta_1, ..., beta_degree]` on the raw
    /// scale, so that `factor[r] = constant + sum of beta_m * values[r]^m`.
    ///
    /// The constant is not returned: it is not a property of this table, having been
    /// moved into the intercept by anchoring.
    ///
    /// Recovered by solving on a basis rescaled to `[-1, 1]` and then expanding back,
    /// rather than fitting powers of the raw values directly. Raw powers of a driver
    /// like age produce a normal matrix with entries spanning many orders of magnitude,
    /// and the recovered coefficients would lose most of their significant digits.
    pub fn variate_coefficients(&self) -> Option<Vec<f64>> {
        let values = self.variate_values()?;
        let degree = self.variate_degree()?;
        let (centre, scale) = variate_basis_params(values)?;

        // Least squares of the factors on [1, u, u^2, ...], solved through the normal
        // equations. The factors lie exactly on the polynomial by construction, so this
        // is a consistent system and the fit is exact.
        let k = degree + 1;
        let mut ata = vec![0.0f64; k * k];
        let mut atb = vec![0.0f64; k];
        for r in 0..values.len() {
            let u = (values[r] - centre) / scale;
            let mut basis = vec![0.0; k];
            let mut p = 1.0;
            for b in basis.iter_mut() {
                *b = p;
                p *= u;
            }
            let f = self.get_rating_factor(r);
            for a in 0..k {
                atb[a] += basis[a] * f;
                for b in 0..k {
                    ata[a * k + b] += basis[a] * basis[b];
                }
            }
        }

        let scaled = crate::glm::solve_spd(&ata, &atb, k)?;

        // Expand sum_m a_m ((v - c)/s)^m into powers of v.
        // beta_j = sum over m >= j of (a_m / s^m) * C(m, j) * (-c)^(m - j)
        let mut raw = vec![0.0f64; degree];
        for m in 1..=degree {
            let a_m = scaled[m] / scale.powi(m as i32);
            for j in 1..=m {
                let binom = binomial(m, j);
                raw[j - 1] += a_m * binom * (-centre).powi((m - j) as i32);
            }
        }
        Some(raw)
    }

    /// Set the table name for better diagnostics
    pub fn with_name(mut self, name: &str) -> Self {
        self.metadata.name = name.to_string();
        self
    }
}

// RatingModel holds multiple RatingTables
#[derive(Clone)]
pub struct RatingModel {
    pub tables: Vec<RatingTable>,
    pub link_function: LinkFunction,
}

impl RatingModel {
    // Private Constructor for internal use
    pub fn new(tables: Vec<RatingTable>, link_function: LinkFunction) -> Self {
        Self {
            tables,
            link_function,
        }
    }

    fn get_link_from_model_json(model_json: &str) -> Result<LinkFunction, serde_json::Error> {
        let model_json: Value = serde_json::from_str(model_json)?;
        let objective = model_json
            .get("objective")
            .and_then(|v| v.as_str())
            .unwrap_or("regression");
        Ok(LinkFunction::from_objective(objective))
    }

    //Constructor method from lgbm json
    pub fn from_lgbm_json(
        model_json: &str,
        consolidation_level: &str,
    ) -> Result<Self, PolarsError> {
        let tables = lgbm_parser::process_lgbm_trees(model_json).map_err(|e| {
            PolarsError::ComputeError(format!("Error processing trees: {}", e).into())
        })?;
        let link_function = Self::get_link_from_model_json(model_json).map_err(|e| {
            PolarsError::ComputeError(format!("Error getting link function: {}", e).into())
        })?;

        match consolidation_level.to_lowercase().as_str() {
            "max" => Ok(lgbm_parser::build_consolidated_tablemodel(
                tables,
                link_function,
            )),
            "analysis" => Ok(lgbm_parser::build_analysis_tablemodel(
                model_json,
                link_function,
            )?),
            _ => Err(PolarsError::ComputeError(
                format!(
                    "Invalid consolidation_level '{}'. Must be 'max' or 'analysis'",
                    consolidation_level
                )
                .into(),
            )),
        }
    }

    // Constructor from a list of DataFrames
    pub fn from_dataframes(
        tables: Vec<DataFrame>,
        link_function: &str,
        feature_columns: Option<Vec<String>>,
        existing_row_number_col: Option<&str>,
    ) -> Result<Self, PolarsError> {
        let link_function = LinkFunction::from_objective(link_function);

        let rating_tables = tables
            .into_iter()
            .map(|df| {
                // If feature columns are specified, select only those plus Rating_Factor
                let filtered_df = if let Some(features) = &feature_columns {
                    let mut cols = features.clone();
                    cols.push("Rating_Factor".to_string());
                    // Also include row_number column if it exists
                    if let Some(row_col) = existing_row_number_col {
                        cols.push(row_col.to_string());
                    }
                    df.select(&cols)
                } else {
                    Ok(df)
                }?;

                Ok(RatingTable::new(filtered_df, existing_row_number_col))
            })
            .collect::<Result<Vec<RatingTable>, PolarsError>>()?;

        Ok(RatingModel::new(rating_tables, link_function))
    }

    pub fn model_tables(&self) -> Vec<DataFrame> {
        self.tables.iter().map(|t| t.data.clone()).collect()
    }

    /// Consolidates all tables in the model (except the mean table) into a minimal set of combined tables
    pub fn consolidate_tables(&self) -> Self {
        // Keep the mean table (first table) separate
        let mean_table = self.tables[0].clone();

        // Combine all other tables
        let combined_tables = consolidation::combine_all_tables(self.tables[1..].to_vec());

        // Create new model with mean table and combined tables
        let mut final_tables = vec![mean_table];
        final_tables.extend(combined_tables);

        RatingModel::new(final_tables, self.link_function.clone())
    }

    /// Predicts a single row of features
    pub fn predict_one(&self, feature_values: &HashMap<String, f64>) -> f64 {
        // Validate features first
        if let Err(err) = self.validate_features(feature_values) {
            panic!("Feature validation failed: {}", err);
        }

        let adjustment: f64 = self
            .tables
            .iter()
            .enumerate()
            .map(|(i, table)| {
                let converted: HashMap<_, _> = feature_values
                    .iter()
                    .filter_map(|(k, &v)| {
                        if let Ok(col) = table.data.column(k) {
                            let feature_value = match col.dtype() {
                                DataType::Int32 => FeatureValue::Categorical(v as i32),
                                DataType::Float64 => FeatureValue::Numeric(v),
                                _ => return None,
                            };
                            Some((k.clone(), feature_value))
                        } else {
                            None
                        }
                    })
                    .collect();
                let pred = table.predict(&converted);
                println!("Table {}: {}", i, pred);
                pred
            })
            .sum();

        self.link_function.inverse(adjustment)
    }

    pub fn predict_linear(&self, df: &DataFrame) -> Result<Vec<f64>, PolarsError> {
        // Predicts values without applying the inverse link function

        // Validate DataFrame columns
        let required_features: HashMap<String, DataType> = self.tables[1..]
            .iter()
            .flat_map(|table| table.get_feature_info())
            .collect();

        // Check for missing columns
        let missing_cols: Vec<_> = required_features
            .keys()
            .filter(|col| {
                !df.get_column_names()
                    .iter()
                    .any(|c| c.as_str() == col.as_str())
            })
            .collect();

        if !missing_cols.is_empty() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Missing required columns: {}",
                    missing_cols
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
                .into(),
            ));
        }

        // Check column types
        for (col, expected_type) in &required_features {
            let df_col = df.column(col)?;
            if df_col.dtype() != expected_type {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Column '{}' has type {:?}, expected {:?}",
                        col,
                        df_col.dtype(),
                        expected_type
                    )
                    .into(),
                ));
            }
        }

        let n_rows = df.height();
        let n_tables = self.tables.len();

        // Thresholds could be tuned based on benchmarking
        const ROW_PARALLEL_THRESHOLD: usize = 10;
        const TABLE_PARALLEL_THRESHOLD: usize = 10;

        let error_holder = Mutex::new(None);

        let predictions: Vec<f64> = match (
            n_rows > ROW_PARALLEL_THRESHOLD,
            n_tables > TABLE_PARALLEL_THRESHOLD,
        ) {
            (true, false) => {
                // Many rows, few tables - parallelize rows only
                (0..n_rows)
                    .into_par_iter()
                    .map(|row_idx| self.predict_row_sequential(df, row_idx, &error_holder))
                    .collect()
            }
            (false, true) => {
                // Few rows, many tables - parallelize tables only
                (0..n_rows)
                    .map(|row_idx| self.predict_row_parallel(df, row_idx, &error_holder))
                    .collect()
            }
            (true, true) => {
                // Many of both - parallelize both
                (0..n_rows)
                    .into_par_iter()
                    .map(|row_idx| self.predict_row_parallel(df, row_idx, &error_holder))
                    .collect()
            }
            (false, false) => {
                // Few of both - no parallelization
                (0..n_rows)
                    .map(|row_idx| self.predict_row_sequential(df, row_idx, &error_holder))
                    .collect()
            }
        };

        Ok(predictions)
    }

    pub fn predict(&self, df: &DataFrame) -> Result<Series, PolarsError> {
        // Predicts values, applying the inverse link function
        let linear_predictions = self.predict_linear(df)?;
        let transformed_predictions = self.apply_link_function(linear_predictions);
        Ok(Series::new("predictions".into(), transformed_predictions))
    }

    pub fn get_link_function(&self) -> String {
        self.link_function.to_string()
    }

    pub fn round_rating_factors(&self, num_decimals: i32) -> RatingModel {
        RatingModel::new(
            self.tables
                .iter()
                .map(|t| t.round_rating_factor(num_decimals))
                .collect(),
            self.link_function.clone(),
        )
    }

    pub fn validate_features(&self, features: &HashMap<String, f64>) -> Result<(), String> {
        // Collect all unique features and their types across all tables
        let mut required_features: HashMap<String, DataType> = HashMap::new();

        for table in &self.tables[1..] {
            // Skip mean table
            for (feat, dtype) in table.get_feature_info() {
                required_features.entry(feat).or_insert(dtype);
            }
        }

        // Check if all required features are present
        let missing_features: Vec<_> = required_features
            .keys()
            .filter(|feat| !features.contains_key(*feat))
            .collect();

        if !missing_features.is_empty() {
            return Err(format!(
                "Missing required features: {}. Required features are: {}",
                missing_features
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                required_features
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        // Validate types (all required features should be convertible to their expected type)
        for (feat, dtype) in &required_features {
            let value = features[feat];
            match dtype {
                DataType::Int32 => {
                    if value.fract() != 0.0 {
                        return Err(format!(
                            "Feature '{}' expects integer values, got {}",
                            feat, value
                        ));
                    }
                }
                DataType::Float64 => {
                    // All f64 values are valid
                    continue;
                }
                _ => {
                    return Err(format!(
                        "Unsupported data type {:?} for feature '{}'",
                        dtype, feat
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn apply_link_function(&self, linear_predictions: Vec<f64>) -> Vec<f64> {
        linear_predictions
            .par_iter()
            .map(|v| self.link_function.inverse(*v))
            .collect::<Vec<f64>>()
    }

    pub fn one_way_analysis(
        &self,
        df: &DataFrame,
        target_column: &str,
        weight_column: Option<&str>,
    ) -> Result<Vec<DataFrame>, PolarsError> {
        crate::analysis::one_way_analysis(self, df, target_column, weight_column)
        // ⭐ NO CLONE
    }

    // NEW: Offset-related methods

    /// Add an offset table to the model (will be used in predictions but not updated by GLM)
    pub fn add_offset_table(&mut self, table: RatingTable) {
        self.tables.push(table.as_offset());
    }

    fn predict_row_sequential(
        &self,
        df: &DataFrame,
        row_idx: usize,
        error_holder: &Mutex<Option<PolarsError>>,
    ) -> f64 {
        if error_holder.lock().unwrap().is_some() {
            return 0.0;
        }

        match self.tables[0].extract_row_features(df, row_idx) {
            Ok(feature_map) => {
                let adjustment: f64 = self
                    .tables
                    .iter()
                    .map(|table| table.predict(&feature_map))
                    .sum();
                adjustment
            }
            Err(e) => {
                handle_error(error_holder, row_idx, e);
                0.0
            }
        }
    }

    fn predict_row_parallel(
        &self,
        df: &DataFrame,
        row_idx: usize,
        error_holder: &Mutex<Option<PolarsError>>,
    ) -> f64 {
        if error_holder.lock().unwrap().is_some() {
            return 0.0;
        }

        match self.tables[0].extract_row_features(df, row_idx) {
            Ok(feature_map) => {
                let adjustment: f64 = self
                    .tables
                    .par_iter()
                    .map(|table| table.predict(&feature_map))
                    .sum();
                adjustment
            }
            Err(e) => {
                handle_error(error_holder, row_idx, e);
                0.0
            }
        }
    }
}

impl Add for RatingModel {
    type Output = Result<Self, PolarsError>;

    fn add(self, other: Self) -> Self::Output {
        self.combine(&other)
    }
}

impl RatingModel {
    /// Combines two RatingModels into a single consolidated model
    pub fn combine(&self, other: &RatingModel) -> Result<Self, PolarsError> {
        // First check that link functions match
        if self.link_function.to_string() != other.link_function.to_string() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot combine models with different link functions: {} and {}",
                    self.link_function.to_string(),
                    other.link_function.to_string()
                )
                .into(),
            ));
        }

        // Sum the mean tables if they exist
        let combined_mean = {
            let mean1 = if !self.tables.is_empty() {
                self.tables[0]
                    .data
                    .column("Rating_Factor")
                    .ok()
                    .and_then(|col| col.f64().ok())
                    .and_then(|series| series.get(0))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let mean2 = if !other.tables.is_empty() {
                other.tables[0]
                    .data
                    .column("Rating_Factor")
                    .ok()
                    .and_then(|col| col.f64().ok())
                    .and_then(|series| series.get(0))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            RatingTable::new(
                DataFrame::new(vec![Series::new(
                    "Rating_Factor".into(),
                    vec![mean1 + mean2],
                )
                .into()])
                .unwrap(),
                None,
            )
        };

        // Start with the combined mean table
        let mut combined_tables = vec![combined_mean];

        // Combine all other tables from both models
        let mut tables_to_combine = Vec::new();
        if self.tables.len() > 1 {
            tables_to_combine.extend(self.tables[1..].iter().cloned());
        }
        if other.tables.len() > 1 {
            tables_to_combine.extend(other.tables[1..].iter().cloned());
        }

        // Use existing combine_all_tables function
        combined_tables.extend(consolidation::combine_all_tables(tables_to_combine));

        Ok(RatingModel::new(
            combined_tables,
            self.link_function.clone(),
        ))
    }

    /// Combines multiple RatingModels into a single consolidated model
    pub fn combine_many(models: Vec<RatingModel>) -> Result<Self, PolarsError> {
        if models.is_empty() {
            return Err(PolarsError::ComputeError("No models to combine".into()));
        }

        let mut iter = models.into_iter();
        let first = iter.next().unwrap();
        iter.try_fold(first, |acc, model| acc + model)
    }
}

fn handle_error(error_holder: &Mutex<Option<PolarsError>>, row_idx: usize, e: PolarsError) {
    let mut error = error_holder.lock().unwrap();
    if error.is_none() {
        *error = Some(PolarsError::ComputeError(
            format!("Error processing row {}: {}", row_idx, e).into(),
        ));
    }
}
