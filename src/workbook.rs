//! The model as a file you can open, edit, and load back.
//!
//! Avenue's opinionated claim is that the tables *are* the model. That only pays off
//! if the tables are a real artifact — something a person can be handed, can change,
//! and can return, with the result meaning exactly what the file says. A representation
//! that only survives inside one process is an implementation detail, not a deliverable.
//!
//! A [`Workbook`] is that artifact: the tables, plus the manifest carrying everything
//! the tables alone cannot say — family and link, which tables are offsets, which rows
//! are locked, which tables are variates, how categorical levels map to codes, and what
//! scale the factors are written on. It saves as one JSON file or as a directory of
//! CSVs, and loads back as the model it was.
//!
//! Three decisions shape it.
//!
//! **The editable artifact is not the review view.** [`crate::report::ModelReport`]
//! renders tables with standard errors, statuses and both scales, for reading. That
//! view deliberately does not round-trip: every extra numeric column would come back as
//! a phantom feature. A workbook carries one factor column and nothing derived, so
//! there is never a question of which number is the model.
//!
//! **The scale is declared rather than guessed.** An actuary edits relativities; the
//! model stores log-scale factors. Writing both would leave two columns encoding one
//! truth, and an edit to the wrong one would be silently ignored. So a workbook writes
//! exactly one, named for what it is, and the manifest records which.
//!
//! The two scales differ in one more way worth knowing: [`Scale::Factor`] round-trips
//! bit for bit, while [`Scale::Relativity`] writes `exp(factor)` and reads back
//! `ln(relativity)`, which agrees to a couple of units in the last place rather than
//! exactly. That is the price of writing the file in the units a person edits, and it
//! is far below anything a rate depends on — but use the factor scale when two models
//! are being compared or a fit reproduced.
//!
//! **Structure is checked when it is loaded, and every fault is reported at once.**
//! An out-of-order band or a missing unbounded row does not fail loudly on its own — it
//! quietly mis-prices, which is the worst failure this library can have. Checking
//! happens at load boundaries rather than in [`crate::rating_model::RatingTable::new`],
//! because fitting reconstructs tables on every sweep and must not pay for it.

use crate::plan::Encoding;
use crate::rating_model::{LinkFunction, RatingModel, RatingTable};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// Bumped when the on-disk shape changes in a way older readers cannot handle.
pub const FORMAT_VERSION: u32 = 1;

/// Factors that lie this far off the variate's own curve are worth remarking on.
const VARIATE_TOLERANCE: f64 = 1e-6;

// ---------------------------------------------------------------- scale

/// Which scale a workbook's factors are written on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scale {
    /// The linear predictor, additive across tables. Column `Rating_Factor`.
    Factor,
    /// `exp(factor)`, multiplicative across tables. Column `Relativity`. Only
    /// meaningful under a log link, and what a pricing actuary edits.
    Relativity,
}

impl Scale {
    /// The column a workbook on this scale carries its factors in.
    pub fn column(&self) -> &'static str {
        match self {
            Scale::Factor => "Rating_Factor",
            Scale::Relativity => "Relativity",
        }
    }

    /// What a log-link model should default to, and what everything else must use.
    pub fn default_for(link: &str) -> Self {
        if link == "log" {
            Scale::Relativity
        } else {
            Scale::Factor
        }
    }

    fn to_factor(&self, value: f64) -> f64 {
        match self {
            Scale::Factor => value,
            Scale::Relativity => value.ln(),
        }
    }

    fn from_factor(&self, factor: f64) -> f64 {
        match self {
            Scale::Factor => factor,
            Scale::Relativity => factor.exp(),
        }
    }
}

// ---------------------------------------------------------------- manifest

/// The variate curve behind a table, so it survives a save.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariateManifest {
    /// What each row is worth on the driver's scale, one per row.
    pub values: Vec<f64>,
    pub degree: usize,
}

/// What one table is, beyond its rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableManifest {
    pub name: String,
    /// File this table lives in, for a CSV workbook.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// The table is fixed: carried in every prediction, never updated by a fit. This
    /// is how an existing rating plan enters a new model.
    #[serde(default)]
    pub is_offset: bool,
    /// Individual rows held fixed while the rest of the table is fitted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locked_rows: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variate: Option<VariateManifest>,
}

/// Everything about a model that its tables cannot say on their own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    /// The Avenue version that wrote this, for provenance.
    pub avenue_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    pub family: String,
    #[serde(default = "default_tweedie")]
    pub tweedie_power: f64,
    /// Derived from the family; recorded so a reader need not know the mapping.
    pub link: String,
    /// Which scale the factor column is on.
    pub scale: Scale,
    /// The response column this model was fitted against, and how exposure entered.
    ///
    /// Recorded so a loaded model can be measured on the same footing as one just
    /// fitted. Without it the tables can score but cannot be validated: nothing would
    /// say which column the predictions are supposed to explain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<String>,
    /// `offset` or `weight`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_role: Option<String>,
    pub tables: Vec<TableManifest>,
    /// Category codes for any string columns, so a level keeps the code it was fitted
    /// with. Without this a workbook could not be scored against fresh data.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub encodings: BTreeMap<String, Vec<(String, i32)>>,
}

fn default_tweedie() -> f64 {
    1.5
}

// ---------------------------------------------------------------- issues

/// Something wrong with a table's structure.
///
/// Every one of these describes a fault that would otherwise mis-price silently, so
/// each names the table, the row where it can be seen, and the repair.
#[derive(Debug, Clone, PartialEq)]
pub struct TableIssue {
    pub table: String,
    pub row: Option<usize>,
    pub code: String,
    pub message: String,
    /// A blocking issue stops the model loading. A non-blocking one is reported and
    /// carried on [`LoadedModel::issues`].
    pub blocking: bool,
}

impl TableIssue {
    fn blocking(table: &str, row: Option<usize>, code: &str, message: String) -> Self {
        Self {
            table: table.to_string(),
            row,
            code: code.to_string(),
            message,
            blocking: true,
        }
    }

    fn note(table: &str, row: Option<usize>, code: &str, message: String) -> Self {
        Self {
            table: table.to_string(),
            row,
            code: code.to_string(),
            message,
            blocking: false,
        }
    }

    /// One line, naming the table and row.
    pub fn describe(&self) -> String {
        match self.row {
            Some(row) => format!(
                "[{}] table '{}' row {}: {}",
                self.code, self.table, row, self.message
            ),
            None => format!("[{}] table '{}': {}", self.code, self.table, self.message),
        }
    }
}

// ---------------------------------------------------------------- validation

/// Check one table's structure.
///
/// `is_intercept` marks table zero, which has different rules: exactly one row and no
/// feature columns.
pub fn check_table(
    df: &DataFrame,
    name: &str,
    factor_column: &str,
    is_intercept: bool,
) -> Vec<TableIssue> {
    let mut issues = Vec::new();

    // ---- the factor column
    let factors = match df.column(factor_column) {
        Err(_) => {
            issues.push(TableIssue::blocking(
                name,
                None,
                "missing_factor_column",
                format!(
                    "No '{}' column. A rating table needs one column of factors; the \
                     columns present are {}.",
                    factor_column,
                    column_list(df)
                ),
            ));
            return issues;
        }
        Ok(column) => match column
            .cast(&DataType::Float64)
            .and_then(|c| c.f64().cloned())
        {
            Ok(values) => values,
            Err(_) => {
                issues.push(TableIssue::blocking(
                    name,
                    None,
                    "factor_column_not_numeric",
                    format!(
                        "Column '{}' has dtype {:?} and cannot be read as a number.",
                        factor_column,
                        column.dtype()
                    ),
                ));
                return issues;
            }
        },
    };

    if df.height() == 0 {
        issues.push(TableIssue::blocking(
            name,
            None,
            "empty_table",
            "The table has no rows, so nothing can match it.".to_string(),
        ));
        return issues;
    }

    for (row, value) in factors.into_iter().enumerate() {
        match value {
            None => issues.push(TableIssue::blocking(
                name,
                Some(row),
                "null_factor",
                format!("'{}' is empty. Every row needs a factor.", factor_column),
            )),
            Some(v) if !v.is_finite() => issues.push(TableIssue::blocking(
                name,
                Some(row),
                "non_finite_factor",
                format!("'{}' is {}, which cannot be scored.", factor_column, v),
            )),
            _ => {}
        }
    }

    // ---- feature columns
    let feature_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|c| c.to_string())
        .filter(|c| c != factor_column)
        .collect();

    if is_intercept {
        if df.height() != 1 {
            issues.push(TableIssue::blocking(
                name,
                None,
                "intercept_not_single_row",
                format!(
                    "The intercept table has {} rows. It must have exactly one — it \
                     applies to every observation.",
                    df.height()
                ),
            ));
        }
        if !feature_names.is_empty() {
            issues.push(TableIssue::blocking(
                name,
                None,
                "intercept_has_features",
                format!(
                    "The intercept table has feature columns ({}). It must carry only \
                     '{}'.",
                    feature_names.join(", "),
                    factor_column
                ),
            ));
        }
        return issues;
    }

    if feature_names.is_empty() {
        issues.push(TableIssue::blocking(
            name,
            None,
            "no_feature_columns",
            format!(
                "The table has no feature columns, so it matches every observation and \
                 is indistinguishable from the intercept. Add the column it rates on."
            ),
        ));
        return issues;
    }

    let mut numeric = Vec::new();
    let mut categorical = Vec::new();
    for feature in &feature_names {
        let column = match df.column(feature) {
            Ok(column) => column,
            Err(_) => continue,
        };
        match column.dtype() {
            DataType::Float64 => numeric.push(feature.clone()),
            DataType::Int32 => categorical.push(feature.clone()),
            other => issues.push(TableIssue::blocking(
                name,
                None,
                "unreadable_dtype",
                format!(
                    "Column '{}' has dtype {:?}. The matcher reads Float64 as a numeric \
                     band and Int32 as a category code, and nothing else — anything else \
                     is dropped, which leaves the column constraining nothing and every \
                     observation matching row 0. Cast it to Float64 or Int32.",
                    feature, other
                ),
            )),
        }
    }

    for feature in &numeric {
        let values = match df.column(feature).and_then(|c| c.f64().cloned()) {
            Ok(values) => values,
            Err(_) => continue,
        };
        if let Some(row) = (0..values.len()).find(|i| values.get(*i).is_none()) {
            issues.push(TableIssue::blocking(
                name,
                Some(row),
                "null_bound",
                format!(
                    "Band bound '{}' is empty. Every row needs an upper bound.",
                    feature
                ),
            ));
            continue;
        }

        // Every numeric column must reach infinity somewhere, or the observations
        // above its largest bound match no row at all and score as NaN.
        let max = values.into_no_null_iter().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_infinite() {
            issues.push(TableIssue::blocking(
                name,
                None,
                "no_unbounded_band",
                format!(
                    "Band bounds '{}' stop at {}. The largest band must be unbounded \
                     (inf), or anything above {} matches no row and scores as NaN. Add a \
                     final row with '{}' = inf.",
                    feature, max, max, feature
                ),
            ));
        }

        // The single-column case is the one people hand-edit, and the one where
        // out-of-order rows silently mis-bin: lookup takes the first row whose bound
        // is not below the value, so an unsorted table answers with the wrong band.
        if numeric.len() == 1 && categorical.is_empty() {
            let bounds: Vec<f64> = values.into_no_null_iter().collect();
            for row in 1..bounds.len() {
                if bounds[row] < bounds[row - 1] {
                    issues.push(TableIssue::blocking(
                        name,
                        Some(row),
                        "bounds_not_ascending",
                        format!(
                            "Band bound '{}' is {} but the row above it is {}. Bounds must \
                             ascend down the table: lookup takes the first row whose bound \
                             is not below the value, so an out-of-order row silently \
                             returns the wrong band. Sort the rows by '{}'.",
                            feature,
                            bounds[row],
                            bounds[row - 1],
                            feature
                        ),
                    ));
                    break;
                }
            }
            for row in 1..bounds.len() {
                if bounds[row] == bounds[row - 1] {
                    issues.push(TableIssue::blocking(
                        name,
                        Some(row),
                        "duplicate_bound",
                        format!(
                            "Band bound '{}' repeats {} from the row above. The second row \
                             can never match, so its factor is dead.",
                            feature, bounds[row]
                        ),
                    ));
                    break;
                }
            }
        }
    }

    for feature in &categorical {
        if let Ok(codes) = df.column(feature).and_then(|c| c.i32().cloned()) {
            if let Some(row) = (0..codes.len()).find(|i| codes.get(*i).is_none()) {
                issues.push(TableIssue::blocking(
                    name,
                    Some(row),
                    "null_level",
                    format!("Level '{}' is empty. Use -999 for a wildcard row.", feature),
                ));
            }
        }
    }

    // A repeated feature tuple means the later row can never win a first-match lookup.
    if let Some(row) = first_duplicate_row(df, &feature_names) {
        issues.push(TableIssue::blocking(
            name,
            Some(row),
            "duplicate_row",
            format!(
                "This row repeats the feature values of an earlier one. Lookup takes the \
                 first match, so this row's factor can never apply."
            ),
        ));
    }

    issues
}

fn column_list(df: &DataFrame) -> String {
    df.get_column_names()
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The first row whose feature values repeat an earlier row's.
fn first_duplicate_row(df: &DataFrame, features: &[String]) -> Option<usize> {
    let mut seen: HashSet<String> = HashSet::new();
    for row in 0..df.height() {
        let key = features
            .iter()
            .map(|feature| match df.column(feature) {
                Ok(column) => match column.get(row) {
                    Ok(value) => value.to_string(),
                    Err(_) => String::new(),
                },
                Err(_) => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\u{1}");
        if !seen.insert(key) {
            return Some(row);
        }
    }
    None
}

// ---------------------------------------------------------------- workbook

/// A model as a portable, editable artifact.
#[derive(Clone)]
pub struct Workbook {
    pub manifest: Manifest,
    /// One frame per table, carrying the feature columns and a single factor column
    /// named by [`Manifest::scale`].
    pub tables: Vec<DataFrame>,
}

impl Workbook {
    /// Build a workbook from a fitted model.
    ///
    /// `scale` defaults to relativities under a log link, which is the scale a pricing
    /// actuary reads and edits.
    #[allow(clippy::too_many_arguments)]
    pub fn from_model(
        model: &RatingModel,
        family: &str,
        table_names: &[String],
        encoding: &Encoding,
        tweedie_power: f64,
        scale: Option<Scale>,
        response: Option<(&str, Option<&str>, Option<&str>)>,
    ) -> Result<Self, PolarsError> {
        let link = model.get_link_function();
        let scale = scale.unwrap_or_else(|| Scale::default_for(&link));
        if scale == Scale::Relativity && link != "log" {
            return Err(PolarsError::ComputeError(
                format!(
                    "Relativities are exp(factor), which only means anything under a log \
                     link; this model's link is '{}'. Write it on the factor scale instead.",
                    link
                )
                .into(),
            ));
        }
        if table_names.len() != model.tables.len() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Got {} table names for {} tables.",
                    table_names.len(),
                    model.tables.len()
                )
                .into(),
            ));
        }

        let mut tables = Vec::with_capacity(model.tables.len());
        let mut manifests = Vec::with_capacity(model.tables.len());

        for (index, table) in model.tables.iter().enumerate() {
            let factors: Vec<f64> = table
                .data
                .column("Rating_Factor")?
                .f64()?
                .into_no_null_iter()
                .collect();

            // Carry the feature columns through untouched, and write exactly one
            // factor column. Nothing derived: a second column encoding the same truth
            // is a second place for an edit to be silently ignored.
            let mut frame = table.data.drop("Rating_Factor")?;
            // Where a column has an encoding, write the level *text*. A file whose
            // region column reads `3` forces the reader to the manifest before they can
            // change anything; one that reads `west` does not. The label is the key,
            // not a second copy of the factor, so this adds no place for an edit to be
            // silently ignored.
            for column in frame
                .get_column_names()
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<String>>()
            {
                let Some(levels) = encoding.maps.get(&column) else {
                    continue;
                };
                let Ok(codes) = frame.column(&column).and_then(|c| c.i32().cloned()) else {
                    continue;
                };
                let labels: Vec<Option<String>> = codes
                    .into_iter()
                    .map(|code| {
                        code.and_then(|c| {
                            levels
                                .iter()
                                .find(|(_, mapped)| *mapped == c)
                                // A code with no level is left as its number rather
                                // than blanked, so nothing is lost.
                                .map(|(text, _)| text.clone())
                                .or_else(|| Some(c.to_string()))
                        })
                    })
                    .collect();
                frame.with_column(Series::new(column.as_str().into(), labels))?;
            }
            frame.with_column(Series::new(
                scale.column().into(),
                factors
                    .iter()
                    .map(|f| scale.from_factor(*f))
                    .collect::<Vec<f64>>(),
            ))?;

            let locked_rows: Vec<usize> = (0..table.data.height())
                .filter(|row| table.is_row_offset(*row))
                .collect();

            manifests.push(TableManifest {
                name: table_names[index].clone(),
                file: Some(csv_file_name(index, &table_names[index])),
                is_offset: table.metadata.is_offset,
                locked_rows,
                variate: table.variate_values().map(|values| VariateManifest {
                    values: values.to_vec(),
                    degree: table.variate_degree().unwrap_or(1),
                }),
            });
            tables.push(frame);
        }

        Ok(Workbook {
            manifest: Manifest {
                format_version: FORMAT_VERSION,
                avenue_version: env!("CARGO_PKG_VERSION").to_string(),
                created: Some(chrono::Utc::now().to_rfc3339()),
                family: family.to_string(),
                tweedie_power,
                link,
                scale,
                target: response.map(|(target, _, _)| target.to_string()),
                exposure: response.and_then(|(_, exposure, _)| exposure.map(str::to_string)),
                exposure_role: response.and_then(|(_, _, role)| role.map(str::to_string)),
                tables: manifests,
                encodings: encoding.maps.clone(),
            },
            tables,
        })
    }

    /// Turn the workbook back into a model, checking every table's structure first.
    ///
    /// The result is an ordinary [`crate::plan::FittedModel`], so a loaded model
    /// scores, validates, reports and saves exactly like one just fitted. Anything
    /// non-blocking noticed on the way lands on its `notes`.
    ///
    /// Every blocking fault is reported together, not just the first: a caller fixing a
    /// hand-edited file should learn everything wrong with it in one pass.
    pub fn to_model(&self) -> Result<crate::plan::FittedModel, PolarsError> {
        if self.manifest.format_version > FORMAT_VERSION {
            return Err(PolarsError::ComputeError(
                format!(
                    "This workbook is format version {}, and this build of Avenue reads up \
                     to version {}. Upgrade avenue_model to open it.",
                    self.manifest.format_version, FORMAT_VERSION
                )
                .into(),
            ));
        }
        if self.manifest.tables.len() != self.tables.len() {
            return Err(PolarsError::ComputeError(
                format!(
                    "The manifest describes {} tables but {} were supplied.",
                    self.manifest.tables.len(),
                    self.tables.len()
                )
                .into(),
            ));
        }
        if self.tables.is_empty() {
            return Err(PolarsError::ComputeError(
                "A workbook needs at least one table, the first being the intercept.".into(),
            ));
        }

        let scale = self.manifest.scale;
        let factor_column = scale.column();
        let mut issues = Vec::new();

        // Level text becomes codes again before anything inspects the frame, so the
        // structural checks see the dtypes the matcher will.
        let mut decoded = Vec::with_capacity(self.tables.len());
        for (index, frame) in self.tables.iter().enumerate() {
            decoded.push(self.decode_levels(
                frame,
                &self.manifest.tables[index].name,
                &mut issues,
            )?);
        }

        for (index, frame) in decoded.iter().enumerate() {
            issues.extend(check_table(
                frame,
                &self.manifest.tables[index].name,
                factor_column,
                index == 0,
            ));
        }

        if scale == Scale::Relativity {
            for (index, frame) in decoded.iter().enumerate() {
                if let Ok(values) = frame.column(factor_column).and_then(|c| c.f64().cloned()) {
                    for (row, value) in values.into_iter().enumerate() {
                        if matches!(value, Some(v) if v <= 0.0) {
                            issues.push(TableIssue::blocking(
                                &self.manifest.tables[index].name,
                                Some(row),
                                "non_positive_relativity",
                                format!(
                                    "Relativity is {}. A relativity is a multiplier and must \
                                     be above zero; its logarithm is the model's factor.",
                                    value.unwrap_or(0.0)
                                ),
                            ));
                        }
                    }
                }
            }
        }

        let blocking: Vec<&TableIssue> = issues.iter().filter(|i| i.blocking).collect();
        if !blocking.is_empty() {
            return Err(PolarsError::ComputeError(
                format!(
                    "This workbook cannot be loaded as a model. {} problem{} found:\n  {}",
                    blocking.len(),
                    if blocking.len() == 1 { "" } else { "s" },
                    blocking
                        .iter()
                        .map(|i| i.describe())
                        .collect::<Vec<_>>()
                        .join("\n  ")
                )
                .into(),
            ));
        }

        // ---- structure is sound; build the tables
        let mut tables = Vec::with_capacity(self.tables.len());
        for (index, frame) in decoded.iter().enumerate() {
            let entry = &self.manifest.tables[index];
            let factors: Vec<f64> = frame
                .column(factor_column)?
                .cast(&DataType::Float64)?
                .f64()?
                .into_no_null_iter()
                .map(|v| scale.to_factor(v))
                .collect();

            let mut data = frame.drop(factor_column)?;
            data.with_column(Series::new("Rating_Factor".into(), factors))?;

            let mut table = RatingTable::new(data, None).with_name(&entry.name);

            if let Some(variate) = &entry.variate {
                if variate.values.len() != table.data.height() {
                    issues.push(TableIssue::note(
                        &entry.name,
                        None,
                        "variate_dropped",
                        format!(
                            "The manifest records a variate with {} values but the table has \
                             {} rows, so the rows were edited without the manifest being \
                             updated. The table is loaded as ordinary free levels; a refit \
                             will estimate every row rather than a degree-{} curve.",
                            variate.values.len(),
                            table.data.height(),
                            variate.degree
                        ),
                    ));
                } else {
                    match table
                        .clone()
                        .as_polynomial_variate(variate.values.clone(), variate.degree)
                    {
                        Ok(with_variate) => {
                            if let Some(row) = off_the_curve(&with_variate) {
                                issues.push(TableIssue::note(
                                    &entry.name,
                                    Some(row),
                                    "variate_factors_edited",
                                    format!(
                                        "The factors no longer lie on the degree-{} curve the \
                                         manifest records. Predictions use the factors as \
                                         written, which is what the table says; a refit will \
                                         pull them back onto a curve.",
                                        variate.degree
                                    ),
                                ));
                            }
                            table = with_variate;
                        }
                        Err(error) => issues.push(TableIssue::note(
                            &entry.name,
                            None,
                            "variate_dropped",
                            format!(
                                "The recorded variate could not be applied ({}), so the table \
                                 is loaded as ordinary free levels.",
                                error
                            ),
                        )),
                    }
                }
            }

            for row in &entry.locked_rows {
                if *row < table.data.height() {
                    table.set_row_offset(*row, true);
                } else {
                    issues.push(TableIssue::note(
                        &entry.name,
                        Some(*row),
                        "locked_row_missing",
                        "The manifest locks a row this table does not have; the lock was \
                         dropped."
                            .to_string(),
                    ));
                }
            }

            if entry.is_offset {
                table = table.as_offset();
            }
            tables.push(table);
        }

        let mut loaded = crate::plan::FittedModel::from_model(
            RatingModel::new(tables, LinkFunction::from_objective(&self.manifest.family)),
            &self.manifest.family,
            self.manifest
                .tables
                .iter()
                .map(|t| t.name.clone())
                .collect(),
            Encoding {
                maps: self.manifest.encodings.clone(),
            },
        );
        loaded.tweedie_power = self.manifest.tweedie_power;
        if let Some(target) = self.manifest.target.as_deref() {
            loaded = loaded.with_response(
                target,
                self.manifest.exposure.as_deref(),
                match self.manifest.exposure_role.as_deref() {
                    Some("offset") => Some(crate::plan::ExposureRole::Offset),
                    Some("weight") => Some(crate::plan::ExposureRole::Weight),
                    _ => None,
                },
            );
        }

        loaded.notes = issues;
        Ok(loaded)
    }
}

impl Workbook {
    /// Turn any level text in `frame` back into the codes it stands for.
    ///
    /// A value that is not a known level but reads as a whole number is taken as a
    /// code, so a file written by hand still works. Anything else is a level the
    /// encoding has never seen, which is reported rather than guessed at: silently
    /// dropping it would leave the row matching nothing.
    fn decode_levels(
        &self,
        frame: &DataFrame,
        table: &str,
        issues: &mut Vec<TableIssue>,
    ) -> Result<DataFrame, PolarsError> {
        let mut out = frame.clone();
        for column in frame
            .get_column_names()
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<String>>()
        {
            let Some(levels) = self.manifest.encodings.get(&column) else {
                continue;
            };
            let Ok(text) = frame.column(&column).and_then(|c| c.str().cloned()) else {
                continue; // already codes
            };
            let codes: Vec<Option<i32>> = text
                .into_iter()
                .enumerate()
                .map(|(row, value)| match value {
                    None => None,
                    Some(value) => {
                        let trimmed = value.trim();
                        match levels.iter().find(|(name, _)| name == trimmed) {
                            Some((_, code)) => Some(*code),
                            None => match trimmed.parse::<i32>() {
                                Ok(code) => Some(code),
                                Err(_) => {
                                    issues.push(TableIssue::blocking(
                                        table,
                                        Some(row),
                                        "unknown_level",
                                        format!(
                                            "'{}' is not a level of '{}'. The manifest knows \
                                             {}. Add it to the manifest's encodings, or \
                                             correct the spelling.",
                                            trimmed,
                                            column,
                                            levels
                                                .iter()
                                                .map(|(name, _)| name.as_str())
                                                .take(20)
                                                .collect::<Vec<_>>()
                                                .join(", ")
                                        ),
                                    ));
                                    None
                                }
                            },
                        }
                    }
                })
                .collect();
            out.with_column(Series::new(column.as_str().into(), codes))?;
        }
        Ok(out)
    }
}

/// The first row whose factor departs from the table's own fitted variate curve.
fn off_the_curve(table: &RatingTable) -> Option<usize> {
    let values = table.variate_values()?;
    let coefficients = table.variate_coefficients()?;
    let factors = table.data.column("Rating_Factor").ok()?.f64().ok()?;
    // The coefficients are recovered from the factors themselves, so a table that is
    // genuinely on a curve reproduces exactly; anything else was edited.
    let base: f64 = coefficients
        .iter()
        .enumerate()
        .map(|(m, c)| c * values[0].powi(m as i32 + 1))
        .sum();
    let first = factors.get(0)?;
    for (row, value) in values.iter().enumerate() {
        let predicted: f64 = coefficients
            .iter()
            .enumerate()
            .map(|(m, c)| c * value.powi(m as i32 + 1))
            .sum::<f64>()
            - base
            + first;
        let actual = factors.get(row)?;
        if (predicted - actual).abs() > VARIATE_TOLERANCE * actual.abs().max(1.0) {
            return Some(row);
        }
    }
    None
}

// ---------------------------------------------------------------- JSON

/// A workbook on disk as one JSON document.
#[derive(Serialize, Deserialize)]
struct JsonWorkbook {
    manifest: Manifest,
    tables: Vec<Vec<BTreeMap<String, serde_json::Value>>>,
}

impl Workbook {
    /// The workbook as one JSON document: manifest and every table, self-contained.
    pub fn to_json(&self) -> Result<String, PolarsError> {
        let tables = self
            .tables
            .iter()
            .map(frame_to_records)
            .collect::<Result<Vec<_>, PolarsError>>()?;
        serde_json::to_string_pretty(&JsonWorkbook {
            manifest: self.manifest.clone(),
            tables,
        })
        .map_err(|e| PolarsError::ComputeError(format!("Could not write workbook: {}", e).into()))
    }

    pub fn from_json(text: &str) -> Result<Self, PolarsError> {
        let parsed: JsonWorkbook = serde_json::from_str(text).map_err(|e| {
            PolarsError::ComputeError(format!("Could not read workbook: {}", e).into())
        })?;
        let tables = parsed
            .tables
            .iter()
            .map(|records| records_to_frame(records))
            .collect::<Result<Vec<_>, PolarsError>>()?;
        Ok(Workbook {
            manifest: parsed.manifest,
            tables,
        })
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), PolarsError> {
        std::fs::write(path.as_ref(), self.to_json()?).map_err(|e| {
            PolarsError::ComputeError(
                format!("Could not write {}: {}", path.as_ref().display(), e).into(),
            )
        })
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, PolarsError> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            PolarsError::ComputeError(
                format!("Could not read {}: {}", path.as_ref().display(), e).into(),
            )
        })?;
        Self::from_json(&text)
    }
}

pub(crate) fn frame_to_records(
    df: &DataFrame,
) -> Result<Vec<BTreeMap<String, serde_json::Value>>, PolarsError> {
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|c| c.to_string())
        .collect();
    let mut out = Vec::with_capacity(df.height());
    for row in 0..df.height() {
        let mut record = BTreeMap::new();
        for name in &names {
            let column = df.column(name)?;
            let value = match column.dtype() {
                // Infinity has no JSON representation, so it travels as the string
                // every rating table means by it.
                DataType::Float64 => match column.f64()?.get(row) {
                    Some(v) if v.is_infinite() => {
                        serde_json::Value::String(if v > 0.0 { "inf" } else { "-inf" }.to_string())
                    }
                    Some(v) => serde_json::json!(v),
                    None => serde_json::Value::Null,
                },
                DataType::Int32 => match column.i32()?.get(row) {
                    Some(v) => serde_json::json!(v),
                    None => serde_json::Value::Null,
                },
                DataType::String => match column.str()?.get(row) {
                    Some(v) => serde_json::Value::String(v.to_string()),
                    None => serde_json::Value::Null,
                },
                _ => serde_json::Value::String(
                    column.get(row).map(|v| v.to_string()).unwrap_or_default(),
                ),
            };
            record.insert(name.clone(), value);
        }
        out.push(record);
    }
    Ok(out)
}

pub(crate) fn records_to_frame(
    records: &[BTreeMap<String, serde_json::Value>],
) -> Result<DataFrame, PolarsError> {
    if records.is_empty() {
        return DataFrame::new(vec![]);
    }
    let names: Vec<String> = records[0].keys().cloned().collect();
    let mut columns: Vec<Column> = Vec::with_capacity(names.len());
    for name in &names {
        let cells: Vec<&serde_json::Value> = records
            .iter()
            .map(|r| r.get(name).unwrap_or(&serde_json::Value::Null))
            .collect();

        // Three shapes, decided by what the values actually are. A column of whole
        // numbers is a category code. A column of numbers, or of the text spellings of
        // infinity, is a numeric band. Anything else is level text — which a workbook
        // writes for every encoded column, so this is the common case rather than an
        // edge one.
        let populated = cells.iter().any(|v| !v.is_null());
        let all_integral = populated
            && cells.iter().all(|v| match v {
                serde_json::Value::Number(n) => n.is_i64() || n.is_u64(),
                serde_json::Value::Null => true,
                _ => false,
            });
        let all_numeric = populated
            && cells.iter().all(|v| match v {
                serde_json::Value::Number(_) | serde_json::Value::Null => true,
                serde_json::Value::String(s) => parse_number(s).is_some(),
                _ => false,
            });

        if all_integral {
            columns.push(
                Series::new(
                    name.as_str().into(),
                    cells
                        .iter()
                        .map(|v| v.as_i64().map(|n| n as i32))
                        .collect::<Vec<Option<i32>>>(),
                )
                .into(),
            );
        } else if all_numeric {
            columns.push(
                Series::new(
                    name.as_str().into(),
                    cells
                        .iter()
                        .map(|v| match v {
                            serde_json::Value::Number(n) => n.as_f64(),
                            serde_json::Value::String(s) => parse_number(s),
                            _ => None,
                        })
                        .collect::<Vec<Option<f64>>>(),
                )
                .into(),
            );
        } else {
            columns.push(
                Series::new(
                    name.as_str().into(),
                    cells
                        .iter()
                        .map(|v| match v {
                            serde_json::Value::String(s) => Some(s.clone()),
                            serde_json::Value::Null => None,
                            other => Some(other.to_string()),
                        })
                        .collect::<Vec<Option<String>>>(),
                )
                .into(),
            );
        }
    }
    DataFrame::new(columns)
}

/// Read a number, accepting the spellings a spreadsheet produces for infinity.
fn parse_number(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "" => None,
        _ => trimmed.parse::<f64>().ok(),
    }
}

// ---------------------------------------------------------------- CSV

fn csv_file_name(index: usize, name: &str) -> String {
    let safe: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    format!("{:02}_{}.csv", index, safe)
}

impl Workbook {
    /// Save as a directory of CSVs plus `manifest.json`.
    ///
    /// This is the form to hand someone who will edit it in a spreadsheet: one file per
    /// rating table, with the factor column named for the scale it is on.
    pub fn save_csv_dir(&self, dir: impl AsRef<Path>) -> Result<(), PolarsError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| {
            PolarsError::ComputeError(format!("Could not create {}: {}", dir.display(), e).into())
        })?;

        let mut manifest = self.manifest.clone();
        for (index, entry) in manifest.tables.iter_mut().enumerate() {
            let file = csv_file_name(index, &entry.name);
            std::fs::write(dir.join(&file), frame_to_csv(&self.tables[index])?).map_err(|e| {
                PolarsError::ComputeError(format!("Could not write {}: {}", file, e).into())
            })?;
            entry.file = Some(file);
        }

        let text = serde_json::to_string_pretty(&manifest).map_err(|e| {
            PolarsError::ComputeError(format!("Could not write manifest: {}", e).into())
        })?;
        std::fs::write(dir.join("manifest.json"), text).map_err(|e| {
            PolarsError::ComputeError(format!("Could not write manifest.json: {}", e).into())
        })
    }

    /// Load a directory written by [`Workbook::save_csv_dir`], after someone has edited
    /// it.
    pub fn load_csv_dir(dir: impl AsRef<Path>) -> Result<Self, PolarsError> {
        let dir = dir.as_ref();
        let text = std::fs::read_to_string(dir.join("manifest.json")).map_err(|e| {
            PolarsError::ComputeError(
                format!(
                    "Could not read {}: {}. A CSV workbook needs the manifest.json written \
                     beside its tables — it carries the family, the scale, which tables are \
                     offsets, and the category codes.",
                    dir.join("manifest.json").display(),
                    e
                )
                .into(),
            )
        })?;
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| {
            PolarsError::ComputeError(format!("Could not read manifest.json: {}", e).into())
        })?;

        let mut tables = Vec::with_capacity(manifest.tables.len());
        for (index, entry) in manifest.tables.iter().enumerate() {
            let file: PathBuf = dir.join(
                entry
                    .file
                    .clone()
                    .unwrap_or_else(|| csv_file_name(index, &entry.name)),
            );
            let text = std::fs::read_to_string(&file).map_err(|e| {
                PolarsError::ComputeError(
                    format!(
                        "Could not read {} for table '{}': {}",
                        file.display(),
                        entry.name,
                        e
                    )
                    .into(),
                )
            })?;
            tables.push(csv_to_frame(&text, &entry.name)?);
        }

        Ok(Workbook { manifest, tables })
    }
}

/// Write a frame as CSV.
///
/// Hand-rolled rather than delegated so infinity round-trips as the literal `inf` a
/// rating table means by it, rather than whatever a writer happens to emit.
fn frame_to_csv(df: &DataFrame) -> Result<String, PolarsError> {
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|c| c.to_string())
        .collect();
    let mut out = String::new();
    out.push_str(&names.join(","));
    out.push('\n');
    for row in 0..df.height() {
        let cells: Vec<String> = names
            .iter()
            .map(|name| -> Result<String, PolarsError> {
                let column = df.column(name)?;
                Ok(match column.dtype() {
                    DataType::Float64 => match column.f64()?.get(row) {
                        Some(v) if v.is_infinite() => {
                            if v > 0.0 {
                                "inf".into()
                            } else {
                                "-inf".into()
                            }
                        }
                        Some(v) => format!("{}", v),
                        None => String::new(),
                    },
                    DataType::Int32 => column
                        .i32()?
                        .get(row)
                        .map(|v| v.to_string())
                        .unwrap_or_default(),
                    DataType::String => escape_csv(column.str()?.get(row).unwrap_or("")),
                    _ => escape_csv(&column.get(row).map(|v| v.to_string()).unwrap_or_default()),
                })
            })
            .collect::<Result<Vec<_>, PolarsError>>()?;
        out.push_str(&cells.join(","));
        out.push('\n');
    }
    Ok(out)
}

fn escape_csv(text: &str) -> String {
    if text.contains(',') || text.contains('"') || text.contains('\n') {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

fn csv_to_frame(text: &str, table: &str) -> Result<DataFrame, PolarsError> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| {
        PolarsError::ComputeError(format!("Table '{}' has no header row.", table).into())
    })?;
    let names: Vec<String> = split_csv(header);

    let mut rows: Vec<Vec<String>> = Vec::new();
    for (index, line) in lines.enumerate() {
        let cells = split_csv(line);
        if cells.len() != names.len() {
            return Err(PolarsError::ComputeError(
                format!(
                    "Table '{}' row {} has {} values but the header has {} columns ({}). A \
                     spreadsheet that added or removed a column will do this.",
                    table,
                    index + 1,
                    cells.len(),
                    names.len(),
                    names.join(", ")
                )
                .into(),
            ));
        }
        rows.push(cells);
    }

    let mut columns: Vec<Column> = Vec::with_capacity(names.len());
    for (position, name) in names.iter().enumerate() {
        let cells: Vec<&str> = rows.iter().map(|r| r[position].as_str()).collect();
        let non_empty: Vec<&str> = cells
            .iter()
            .copied()
            .filter(|c| !c.trim().is_empty())
            .collect();

        // A column of whole numbers is a category code; a column with a decimal point,
        // an exponent or an infinity is a numeric band.
        let integral =
            !non_empty.is_empty() && non_empty.iter().all(|c| c.trim().parse::<i32>().is_ok());
        if integral {
            columns.push(
                Series::new(
                    name.as_str().into(),
                    cells
                        .iter()
                        .map(|c| c.trim().parse::<i32>().ok())
                        .collect::<Vec<Option<i32>>>(),
                )
                .into(),
            );
            continue;
        }

        let numeric = !non_empty.is_empty() && non_empty.iter().all(|c| parse_number(c).is_some());
        if numeric {
            columns.push(
                Series::new(
                    name.as_str().into(),
                    cells
                        .iter()
                        .map(|c| parse_number(c))
                        .collect::<Vec<Option<f64>>>(),
                )
                .into(),
            );
        } else {
            columns.push(
                Series::new(
                    name.as_str().into(),
                    cells.iter().map(|c| c.to_string()).collect::<Vec<String>>(),
                )
                .into(),
            );
        }
    }
    DataFrame::new(columns)
}

/// Split one CSV line, honouring quoted fields.
fn split_csv(line: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                current.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => cells.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    cells.push(current);
    cells.into_iter().map(|c| c.trim().to_string()).collect()
}
