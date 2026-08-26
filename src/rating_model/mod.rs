use polars::prelude::*;
use polars::frame::DataFrame;
use polars::series::IntoSeries;
use std::collections::HashMap;
use serde_json::Value;
use std::ops::Add;
use rayon::prelude::*;
use std::sync::Mutex;
use polars::error::PolarsError;

// Internal modules
mod lgbm_parser;
mod consolidation;

// Re-export public functions from lgbm_parser
pub use lgbm_parser::{process_lgbm_trees, build_analysis_tablemodel, build_consolidated_tablemodel};

// Re-export public functions from consolidation
pub use consolidation::{expand_and_combine_tables, combine_all_tables};

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
    /// The row factors are constrained to a straight line through a value attached to
    /// each row: `factor[r] = slope * values[r]`, up to the constant the intercept
    /// absorbs. A five-row table spends **one** parameter, whatever its row count.
    ///
    /// This is the classical actuarial *variate*: age entered as a continuous driver
    /// rather than as a set of independent levels. Three things follow from it that
    /// free levels do not give you:
    ///
    /// * The fitted curve is smooth and monotone by construction, not by penalty.
    /// * Rows with little or no exposure still get a sensible factor, read off the
    ///   line rather than left at their starting value.
    /// * The table is still an ordinary step table, so it deploys unchanged.
    ///
    /// `values` is one number per row: what that row is worth on the driver's scale.
    /// It is supplied rather than derived from the table's own numeric column, because
    /// that column holds inclusive bin *upper bounds* — the top bin's bound is normally
    /// infinite, and a bound is the edge of a bin rather than a point inside it. See
    /// [`RatingTable::as_variate`].
    Variate { values: Vec<f64> },
}

impl Default for TableSemantics {
    fn default() -> Self {
        TableSemantics::Step
    }
}

/// Metadata for a RatingTable
#[derive(Debug, Clone)]
pub struct TableMetadata {
    pub name: String,
    pub is_offset: bool,      // Table is fixed, not updated by GLM
    pub is_updatable: bool,   // Can GLM update this table's factors?
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
    pub is_offset: bool,  // Row is locked, not updated by GLM
}

impl Default for RowMetadata {
    fn default() -> Self {
        Self {
            is_offset: false,
        }
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
    Identity,    // for 'regression'
    Logit,      // for 'binary'
    Log,        // for 'poisson', 'gamma', 'tweedie'
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
    numeric_columns: HashMap<String, usize>,    // column name -> index
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
                },
                DataType::Int32 => {
                    categorical_columns.insert(col_name.to_string(), idx);
                },
                _ => continue,
            }
        }

        Self {
            data:data.clone(),
            numeric_columns,
            categorical_columns,
            metadata: TableMetadata::default(),
            row_metadata: None,  // Lazy initialization - only create if needed
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
        let categorical_columns_vec: Vec<_> = self.categorical_columns.iter()
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
        let numeric_columns_vec: Vec<_> = self.numeric_columns.iter()
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
                    let col_val = unsafe {
                        categorical_columns_vec[idx].get_unchecked(i)
                    };

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
                    let col_val = unsafe {
                        numeric_columns_vec[idx].get_unchecked(i)
                    };

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
            rating_col.f64().unwrap().get_unchecked(row).unwrap_or(f64::NAN)
        }
    }

    pub fn predict(&self, feature_values: &HashMap<String, FeatureValue>) -> f64 {
        // Matches a row to the feature values and returns the rating factor, does not apply the link function
        match self.find_row_match(feature_values) {
            Some(row) => {
                // Get rating factor from the matching row
                self.get_rating_factor(row)
            },
            None => f64::NAN
        }
    }

    pub fn get_rating_factor_batch(&self, row_nums: Series) -> Result<Vec<f64>, PolarsError> {
        let mut df = self.data.clone();

        // Create temporary row numbers for joining
        let n_rows = df.height();
        let temp_row_numbers = Series::new("row_number".into(), (0..n_rows).map(|x| x as i32).collect::<Vec<i32>>());
        let df_with_rownums = df.with_column(temp_row_numbers)?;

        let mut row_number = row_nums.clone();
        row_number.rename("row_number".into());
        let out_df = DataFrame::new(vec![row_number.into()])?;

        // join using temporary row numbers
        let joined = out_df.join(&df_with_rownums, ["row_number"], ["row_number"], JoinArgs::new(JoinType::Left), None)?;
        Ok(joined.column("Rating_Factor")?.f64()?.into_iter().flatten().collect())
    }

    pub fn predict_batch(&self, df: &DataFrame) -> Vec<f64> {
        // Predicts a batch of rows, does not apply the link function
        // Uses parallel processing if the number of rows is greater than the ROW_PARALLEL_THRESHOLD
        // Otherwise, it processes rows sequentially
        const ROW_PARALLEL_THRESHOLD: usize = 10;
        let n_rows = df.height();
        if n_rows > ROW_PARALLEL_THRESHOLD {
            (0..n_rows).into_par_iter()
                .map(|row_idx| {
                    match self.extract_row_features(df, row_idx) {
                        Ok(features) => self.predict(&features),
                        Err(_) => f64::NAN,
                    }
                })
                .collect()
        } else {
            (0..n_rows)
                .map(|row_idx| {
                    match self.extract_row_features(df, row_idx) {
                        Ok(features) => self.predict(&features),
                        Err(_) => f64::NAN,
                    }
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
                Some(val) => Some((val * 10f64.powi(num_decimals)).round() / 10f64.powi(num_decimals)),
                None => None
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
        self.numeric_columns.keys().all(|k| feature_values.contains_key(k)) &&
        self.categorical_columns.keys().all(|k| feature_values.contains_key(k))
    }

    // Add new method to get feature info
    pub fn get_feature_info(&self) -> HashMap<String, DataType> {
        self.data.get_column_names().iter()
            .filter(|&name| *name != "Rating_Factor")
            .map(|name| {
                (name.to_string(), self.data.column(name).unwrap().dtype().clone())
            })
            .collect()
    }

    // Add extract_row_features method to RatingTable
    fn extract_row_features(&self, df: &DataFrame, row_idx: usize) -> Result<HashMap<String, FeatureValue>, PolarsError> {
        let mut feature_values = HashMap::new();

        for col_name in df.get_column_names() {
            let column = df.column(col_name)?;
            match column.dtype() {
                DataType::Float64 => {
                    if let Some(value) = column.f64()?.get(row_idx) {
                        feature_values.insert(col_name.to_string(), FeatureValue::Numeric(value));
                    }
                },
                DataType::Int32 => {
                    if let Some(value) = column.i32()?.get(row_idx) {
                        feature_values.insert(col_name.to_string(), FeatureValue::Categorical(value));
                    }
                },
                _ => continue,
            }
        }

        Ok(feature_values)
    }

    pub fn one_way_analysis_table(
        &self,
        df: &DataFrame,  // ⭐ REFERENCE
        target_column: &str,
        weight_column: Option<&str>
    ) -> Result<DataFrame, PolarsError> {
        crate::analysis::one_way_analysis_table(self, df, target_column, weight_column)  // ⭐ NO CLONE
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
            self.row_metadata = Some(vec![
                RowMetadata::default();
                self.data.height()
            ]);
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
    pub fn as_variate(mut self, values: Vec<f64>) -> Result<Self, PolarsError> {
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
                    label, values.len(), n_rows
                ).into(),
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
                    ).into(),
                ));
            }
        }

        let distinct = values.iter().any(|v| *v != values[0]);
        if !distinct {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot make {} a variate: all {} values are {}. A line through \
                     identical points has no slope to estimate.",
                    label, n_rows, values[0]
                ).into(),
            ));
        }

        if (0..n_rows).any(|r| self.is_row_offset(r)) {
            return Err(PolarsError::ComputeError(
                format!(
                    "Cannot make {} a variate: it has locked rows. Every factor is derived \
                     from the one slope, so pinning a single row would break the line. Lock \
                     the whole table with as_offset() instead.",
                    label
                ).into(),
            ));
        }

        self.metadata.semantics = TableSemantics::Variate { values };
        Ok(self)
    }

    /// How many free parameters this table's rows represent.
    pub fn semantics(&self) -> &TableSemantics {
        &self.metadata.semantics
    }

    /// The per-row values behind a variate table, or `None` for a step table.
    pub fn variate_values(&self) -> Option<&[f64]> {
        match &self.metadata.semantics {
            TableSemantics::Variate { values } => Some(values),
            TableSemantics::Step => None,
        }
    }

    /// The fitted slope of a variate table, recovered from any two rows whose values
    /// differ. `None` for a step table.
    pub fn variate_slope(&self) -> Option<f64> {
        let values = self.variate_values()?;
        let v0 = values[0];
        let r = values.iter().position(|v| *v != v0)?;
        Some((self.get_rating_factor(r) - self.get_rating_factor(0)) / (values[r] - v0))
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
        let objective = model_json.get("objective")
            .and_then(|v| v.as_str())
            .unwrap_or("regression");
        Ok(LinkFunction::from_objective(objective))
    }

    //Constructor method from lgbm json
    pub fn from_lgbm_json(model_json: &str, consolidation_level: &str) -> Result<Self, PolarsError> {
        let tables = lgbm_parser::process_lgbm_trees(model_json).map_err(|e| PolarsError::ComputeError(format!("Error processing trees: {}", e).into()))?;
        let link_function = Self::get_link_from_model_json(model_json).map_err(|e| PolarsError::ComputeError(format!("Error getting link function: {}", e).into()))?;

        match consolidation_level.to_lowercase().as_str() {
            "max" => Ok(lgbm_parser::build_consolidated_tablemodel(tables, link_function)),
            "analysis" => Ok(lgbm_parser::build_analysis_tablemodel(model_json, link_function)?),
            _ => Err(PolarsError::ComputeError(
                format!("Invalid consolidation_level '{}'. Must be 'max' or 'analysis'",
                    consolidation_level).into()
            ))
        }
    }

    // Constructor from a list of DataFrames
    pub fn from_dataframes(
        tables: Vec<DataFrame>,
        link_function: &str,
        feature_columns: Option<Vec<String>>,
        existing_row_number_col: Option<&str>
    ) -> Result<Self, PolarsError> {

        let link_function = LinkFunction::from_objective(link_function);

        let rating_tables = tables.into_iter()
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

        let adjustment: f64 = self.tables.iter().enumerate()
            .map(|(i,table)| {
                let converted: HashMap<_, _> = feature_values.iter()
                    .filter_map(|(k, &v)| {
                        if let Ok(col) = table.data.column(k) {
                            let feature_value = match col.dtype() {
                                DataType::Int32 => FeatureValue::Categorical(v as i32),
                                DataType::Float64 => FeatureValue::Numeric(v),
                                _ => return None
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
        let required_features: HashMap<String, DataType> = self.tables[1..].iter()
            .flat_map(|table| table.get_feature_info())
            .collect();

        // Check for missing columns
        let missing_cols: Vec<_> = required_features.keys()
            .filter(|col| !df.get_column_names().iter().any(|c| c.as_str() == col.as_str()))
            .collect();

        if !missing_cols.is_empty() {
            return Err(PolarsError::ComputeError(
                format!("Missing required columns: {}", missing_cols.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")).into()
            ));
        }

        // Check column types
        for (col, expected_type) in &required_features {
            let df_col = df.column(col)?;
            if df_col.dtype() != expected_type {
                return Err(PolarsError::ComputeError(
                    format!("Column '{}' has type {:?}, expected {:?}",
                        col, df_col.dtype(), expected_type).into()
                ));
            }
        }

        let n_rows = df.height();
        let n_tables = self.tables.len();

        // Thresholds could be tuned based on benchmarking
        const ROW_PARALLEL_THRESHOLD: usize = 10;
        const TABLE_PARALLEL_THRESHOLD: usize = 10;

        let error_holder = Mutex::new(None);

        let predictions: Vec<f64> = match (n_rows > ROW_PARALLEL_THRESHOLD, n_tables > TABLE_PARALLEL_THRESHOLD) {
            (true, false) => {
                // Many rows, few tables - parallelize rows only
                (0..n_rows).into_par_iter()
                    .map(|row_idx| self.predict_row_sequential(df, row_idx, &error_holder))
                    .collect()
            },
            (false, true) => {
                // Few rows, many tables - parallelize tables only
                (0..n_rows).map(|row_idx| {
                    self.predict_row_parallel(df, row_idx, &error_holder)
                }).collect()
            },
            (true, true) => {
                // Many of both - parallelize both
                (0..n_rows).into_par_iter()
                    .map(|row_idx| self.predict_row_parallel(df, row_idx, &error_holder))
                    .collect()
            },
            (false, false) => {
                // Few of both - no parallelization
                (0..n_rows).map(|row_idx| {
                    self.predict_row_sequential(df, row_idx, &error_holder)
                }).collect()
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
        RatingModel::new(self.tables.iter().map(|t| t.round_rating_factor(num_decimals)).collect(), self.link_function.clone())
    }

    pub fn validate_features(&self, features: &HashMap<String, f64>) -> Result<(), String> {
        // Collect all unique features and their types across all tables
        let mut required_features: HashMap<String, DataType> = HashMap::new();

        for table in &self.tables[1..] { // Skip mean table
            for (feat, dtype) in table.get_feature_info() {
                required_features.entry(feat).or_insert(dtype);
            }
        }

        // Check if all required features are present
        let missing_features: Vec<_> = required_features.keys()
            .filter(|feat| !features.contains_key(*feat))
            .collect();

        if !missing_features.is_empty() {
            return Err(format!(
                "Missing required features: {}. Required features are: {}",
                missing_features.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
                required_features.keys().cloned().collect::<Vec<_>>().join(", ")
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
                },
                DataType::Float64 => {
                    // All f64 values are valid
                    continue;
                },
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
        linear_predictions.par_iter().map(|v| self.link_function.inverse(*v)).collect::<Vec<f64>>()
    }

    pub fn one_way_analysis(&self, df: &DataFrame, target_column: &str, weight_column: Option<&str>) -> Result<Vec<DataFrame>, PolarsError> {
        crate::analysis::one_way_analysis(self, df, target_column, weight_column)  // ⭐ NO CLONE
    }

    // NEW: Offset-related methods

    /// Add an offset table to the model (will be used in predictions but not updated by GLM)
    pub fn add_offset_table(&mut self, table: RatingTable) {
        self.tables.push(table.as_offset());
    }

    fn predict_row_sequential(&self, df: &DataFrame, row_idx: usize,
                            error_holder: &Mutex<Option<PolarsError>>) -> f64 {
        if error_holder.lock().unwrap().is_some() {
            return 0.0;
        }

        match self.tables[0].extract_row_features(df, row_idx) {
            Ok(feature_map) => {
                let adjustment: f64 = self.tables.iter()
                    .map(|table| table.predict(&feature_map))
                    .sum();
                adjustment
            },
            Err(e) => {
                handle_error(error_holder, row_idx, e);
                0.0
            }
        }
    }

    fn predict_row_parallel(&self, df: &DataFrame, row_idx: usize,
                          error_holder: &Mutex<Option<PolarsError>>) -> f64 {
        if error_holder.lock().unwrap().is_some() {
            return 0.0;
        }

        match self.tables[0].extract_row_features(df, row_idx) {
            Ok(feature_map) => {
                let adjustment: f64 = self.tables.par_iter()
                    .map(|table| table.predict(&feature_map))
                    .sum();
                adjustment
            },
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
                format!("Cannot combine models with different link functions: {} and {}",
                    self.link_function.to_string(),
                    other.link_function.to_string()
                ).into()
            ));
        }

        // Sum the mean tables if they exist
        let combined_mean = {
            let mean1 = if !self.tables.is_empty() {
                self.tables[0].data
                    .column("Rating_Factor")
                    .ok()
                    .and_then(|col| col.f64().ok())
                    .and_then(|series| series.get(0))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let mean2 = if !other.tables.is_empty() {
                other.tables[0].data
                    .column("Rating_Factor")
                    .ok()
                    .and_then(|col| col.f64().ok())
                    .and_then(|series| series.get(0))
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            RatingTable ::new(
                DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![mean1 + mean2]).into()]).unwrap(),
                None
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

        Ok(RatingModel::new(combined_tables, self.link_function.clone()))
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
            format!("Error processing row {}: {}", row_idx, e).into()
        ));
    }
}
