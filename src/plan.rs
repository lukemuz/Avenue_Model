//! Declarative model plans: say what the model *is*, and let the plan build it.
//!
//! Hand-building rating tables means holding conventions in your head — bands are
//! inclusive upper bounds, ascending, ending at infinity; categoricals are `Int32`
//! codes; table zero is the intercept; `Rating_Factor` is on the linear-predictor
//! scale. A caller who gets one of those wrong does not get an obviously broken
//! model, they get a plausible one.
//!
//! A [`Plan`] states the model instead:
//!
//! ```ignore
//! let plan = Plan::frequency("exposure")
//!     .with(Term::banded("driver_age", Breaks::Explicit(vec![21.0, 25.0, 35.0, 50.0, 70.0])))
//!     .with(Term::categorical("region"))
//!     .with(Term::variate("vehicle_value", Breaks::Quantile(10), 2));
//!
//! let check = plan.check(&df, "claim_count")?;   // before fitting anything
//! let fitted = plan.fit(&df, "claim_count", options)?;
//! ```
//!
//! Two properties matter more than the brevity:
//!
//! **The plan is data.** It round-trips through JSON, so it can be saved, diffed,
//! shown to a person for approval, edited, and re-run. It is the model's source code,
//! where previously the intent was spread across the DataFrame construction that
//! happened to build the tables.
//!
//! **Nothing is decided silently.** [`Plan::check`] reports what the plan *would*
//! do — the band edges a quantile rule picked, the base level chosen, how many levels
//! each factor carries — alongside everything wrong with the data, all before a fit is
//! burned. Every default the plan applies is in [`PlanCheck::resolved`] to be read
//! back and relayed.

use crate::glm::{fit_glm_with_diagnostics, GLMDiagnostics, GLMOptions};
use crate::rating_model::{RatingModel, RatingTable};
use crate::validation::{validate, Severity, Validation, ValidationOptions};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Levels above this in a single categorical factor are worth remarking on.
const WIDE_FACTOR: usize = 50;
/// Total rows above this in one table are worth remarking on.
const WIDE_TABLE: usize = 5_000;

// ---------------------------------------------------------------- specification

/// How a numeric driver is cut into bands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Breaks {
    /// Cut points given outright. These are the inclusive upper bounds of all but the
    /// last band, so `[21, 25, 35]` produces four bands: up to 21, 21 to 25, 25 to 35,
    /// and everything above 35. A final unbounded band is always added, because an
    /// observation that matches no row would otherwise be dropped.
    Explicit { edges: Vec<f64> },
    /// `n` bands of roughly equal count, cut at the sample quantiles. Duplicate edges
    /// are collapsed, so a column with heavy ties yields fewer bands than asked.
    Quantile { n: usize },
    /// `n` bands of equal width between the column's minimum and maximum.
    EqualWidth { n: usize },
}

impl Breaks {
    pub fn explicit(edges: Vec<f64>) -> Self {
        Breaks::Explicit { edges }
    }
    pub fn quantile(n: usize) -> Self {
        Breaks::Quantile { n }
    }
    pub fn equal_width(n: usize) -> Self {
        Breaks::EqualWidth { n }
    }
}

/// Which level a categorical factor is anchored on.
///
/// The base level's factor is fixed at zero under the default anchoring, so every
/// other level reads as a relativity against it. It only applies to categorical
/// factors: a banded table's rows must ascend for the matcher's binary search, so its
/// base is necessarily the lowest band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Base {
    /// The lowest code, or the first level alphabetically for a string column.
    First,
    /// The level carrying the most exposure. Usually what you want: the base level is
    /// the one every relativity is quoted against, so a thin one makes every other
    /// level's standard error larger than it needs to be.
    MostExposed,
    /// A named level, given as it appears in the data.
    Level { value: String },
}

impl Default for Base {
    fn default() -> Self {
        Base::MostExposed
    }
}

/// One factor in the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Term {
    /// A numeric driver cut into bands, each band carrying its own free factor.
    Banded { column: String, breaks: Breaks },
    /// A categorical driver, one free factor per level.
    Categorical {
        column: String,
        #[serde(default)]
        base: Base,
    },
    /// A numeric driver whose band factors are tied to a polynomial, so the table
    /// costs `degree` parameters however many bands it has. Deploys as an ordinary
    /// step table.
    Variate {
        column: String,
        breaks: Breaks,
        /// What each band is worth on the driver's scale, one per band. Defaults to
        /// the midpoint of each band, with the open-ended top band taking its lower
        /// edge plus the previous band's width.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        values: Option<Vec<f64>>,
        degree: usize,
    },
    /// Several drivers crossed into one table, so the factors are free to differ in
    /// every combination. `breaks` is positional: `Some` bands that column, `None`
    /// treats it as categorical.
    Interaction {
        columns: Vec<String>,
        breaks: Vec<Option<Breaks>>,
    },
}

impl Term {
    pub fn banded(column: &str, breaks: Breaks) -> Self {
        Term::Banded {
            column: column.to_string(),
            breaks,
        }
    }
    pub fn categorical(column: &str) -> Self {
        Term::Categorical {
            column: column.to_string(),
            base: Base::default(),
        }
    }
    pub fn categorical_based_on(column: &str, base: Base) -> Self {
        Term::Categorical {
            column: column.to_string(),
            base,
        }
    }
    pub fn variate(column: &str, breaks: Breaks, degree: usize) -> Self {
        Term::Variate {
            column: column.to_string(),
            breaks,
            values: None,
            degree,
        }
    }
    pub fn interaction(columns: Vec<&str>, breaks: Vec<Option<Breaks>>) -> Self {
        Term::Interaction {
            columns: columns.into_iter().map(str::to_string).collect(),
            breaks,
        }
    }

    /// Every data column this term reads.
    pub fn columns(&self) -> Vec<&str> {
        match self {
            Term::Banded { column, .. }
            | Term::Categorical { column, .. }
            | Term::Variate { column, .. } => vec![column.as_str()],
            Term::Interaction { columns, .. } => columns.iter().map(String::as_str).collect(),
        }
    }

    /// The table name this term produces.
    pub fn name(&self) -> String {
        match self {
            Term::Banded { column, .. }
            | Term::Categorical { column, .. }
            | Term::Variate { column, .. } => column.clone(),
            Term::Interaction { columns, .. } => columns.join(" x "),
        }
    }
}

/// How an exposure column enters the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureRole {
    /// `log(exposure)` is added to the linear predictor and held fixed. The standard
    /// idiom for counts: the fitted factors are then rates, and the target is a count.
    Offset,
    /// Exposure is a prior weight. The target is a rate or an average.
    Weight,
}

/// A model, stated rather than constructed.
///
/// Round-trips through JSON so it can be saved, diffed and re-run. Build one with
/// [`Plan::frequency`], [`Plan::severity`], [`Plan::pure_premium`] or [`Plan::new`],
/// then add terms with [`Plan::with`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    /// One of `gaussian`, `poisson`, `gamma`, `tweedie`, `binomial`.
    pub family: String,
    #[serde(default = "default_tweedie")]
    pub tweedie_power: f64,
    /// Exposure column, if the model has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<String>,
    /// Whether that exposure is an offset or a weight. Defaults to `Offset` for
    /// Poisson and `Weight` otherwise, which are the standard idioms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_role: Option<ExposureRole>,
    pub terms: Vec<Term>,
}

fn default_tweedie() -> f64 {
    1.5
}

impl Plan {
    pub fn new(family: &str) -> Self {
        Self {
            family: family.to_string(),
            tweedie_power: 1.5,
            exposure: None,
            exposure_role: None,
            terms: Vec::new(),
        }
    }

    /// Claim counts: Poisson with `log(exposure)` as an offset, so factors are rates.
    pub fn frequency(exposure: &str) -> Self {
        Self {
            exposure: Some(exposure.to_string()),
            exposure_role: Some(ExposureRole::Offset),
            ..Self::new("poisson")
        }
    }

    /// Claim size: Gamma, weighted by claim count.
    pub fn severity(claim_count: &str) -> Self {
        Self {
            exposure: Some(claim_count.to_string()),
            exposure_role: Some(ExposureRole::Weight),
            ..Self::new("gamma")
        }
    }

    /// Loss per unit exposure: Tweedie, weighted by exposure.
    pub fn pure_premium(exposure: &str) -> Self {
        Self {
            exposure: Some(exposure.to_string()),
            exposure_role: Some(ExposureRole::Weight),
            ..Self::new("tweedie")
        }
    }

    pub fn with(mut self, term: Term) -> Self {
        self.terms.push(term);
        self
    }

    pub fn with_tweedie_power(mut self, power: f64) -> Self {
        self.tweedie_power = power;
        self
    }

    /// How exposure enters, resolving the family default when unset.
    pub fn resolved_exposure_role(&self) -> ExposureRole {
        self.exposure_role.unwrap_or({
            if self.family.eq_ignore_ascii_case("poisson") {
                ExposureRole::Offset
            } else {
                ExposureRole::Weight
            }
        })
    }

    pub fn to_json(&self) -> Result<String, PolarsError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PolarsError::ComputeError(format!("Could not serialise plan: {}", e).into()))
    }

    pub fn from_json(text: &str) -> Result<Self, PolarsError> {
        serde_json::from_str(text)
            .map_err(|e| PolarsError::ComputeError(format!("Could not read plan: {}", e).into()))
    }
}

// ---------------------------------------------------------------- encoding

/// Category codes for the string columns a plan uses.
///
/// Carried alongside the fitted model so scoring assigns the same code to the same
/// level. Deriving codes from the data each time would silently shift them whenever a
/// level was missing from a batch.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Encoding {
    /// Column to (level text, code), codes assigned in sorted order of the text.
    pub maps: BTreeMap<String, Vec<(String, i32)>>,
}

impl Encoding {
    fn code_for(&self, column: &str, value: &str) -> Option<i32> {
        self.maps
            .get(column)?
            .iter()
            .find(|(text, _)| text == value)
            .map(|(_, code)| *code)
    }

    /// The level text a code stands for, for presentation.
    pub fn label_for(&self, column: &str, code: i32) -> Option<&str> {
        self.maps
            .get(column)?
            .iter()
            .find(|(_, c)| *c == code)
            .map(|(text, _)| text.as_str())
    }

    pub fn is_encoded(&self, column: &str) -> bool {
        self.maps.contains_key(column)
    }
}

/// A code that no level maps to, so an unseen string fails to match and is reported
/// rather than colliding with a real level.
const UNSEEN_CODE: i32 = i32::MIN;

/// Data with the dtypes the matcher reads, plus any derived offset column.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub df: DataFrame,
    pub encoding: Encoding,
    pub weight_col: Option<String>,
    pub offset_col: Option<String>,
}

/// Name of the offset column a plan derives from exposure.
pub const DERIVED_OFFSET: &str = "__avenue_log_exposure";

impl Plan {
    /// Cast the columns this plan uses into the dtypes the matcher reads, and derive
    /// the offset column when exposure enters that way.
    ///
    /// Pass the [`Encoding`] from fitting when preparing data to score, so string
    /// levels keep the codes they were fitted with.
    pub fn prepare(
        &self,
        df: &DataFrame,
        encoding: Option<&Encoding>,
    ) -> Result<Prepared, PolarsError> {
        let mut out = df.clone();
        let mut built = Encoding::default();

        for term in &self.terms {
            let numeric_columns: HashSet<&str> = match term {
                Term::Banded { column, .. } | Term::Variate { column, .. } => {
                    [column.as_str()].into_iter().collect()
                }
                Term::Categorical { .. } => HashSet::new(),
                Term::Interaction { columns, breaks } => columns
                    .iter()
                    .zip(breaks.iter())
                    .filter(|(_, b)| b.is_some())
                    .map(|(c, _)| c.as_str())
                    .collect(),
            };

            for column in term.columns() {
                let series = out.column(column).map_err(|_| missing_column(column, df))?;
                if numeric_columns.contains(column) {
                    let cast = series.cast(&DataType::Float64).map_err(|_| {
                        PolarsError::ComputeError(
                            format!(
                                "Column '{}' has dtype {:?} and is used as a banded numeric \
                                 driver, but it cannot be read as a number.",
                                column,
                                series.dtype()
                            )
                            .into(),
                        )
                    })?;
                    out.with_column(cast)?;
                } else {
                    let (cast, map) = encode_categorical(series, column, encoding)?;
                    if let Some(map) = map {
                        built.maps.insert(column.to_string(), map);
                    }
                    out.with_column(cast)?;
                }
            }
        }

        // An encoding supplied by the caller is authoritative; carry it forward whole
        // so a column absent from this batch keeps its codes.
        let encoding = match encoding {
            Some(existing) => existing.clone(),
            None => built,
        };

        let mut weight_col = None;
        let mut offset_col = None;
        if let Some(exposure) = &self.exposure {
            let series = out
                .column(exposure)
                .map_err(|_| missing_column(exposure, df))?
                .cast(&DataType::Float64)
                .map_err(|_| {
                    PolarsError::ComputeError(
                        format!("Exposure column '{}' cannot be read as a number.", exposure).into(),
                    )
                })?;
            out.with_column(series.clone())?;
            match self.resolved_exposure_role() {
                ExposureRole::Weight => weight_col = Some(exposure.clone()),
                ExposureRole::Offset => {
                    let values = series.f64()?;
                    let logs: Vec<f64> = values
                        .into_iter()
                        .map(|v| match v {
                            Some(x) if x > 0.0 => x.ln(),
                            // Zero or missing exposure contributes nothing; it is
                            // reported by `check` rather than silently logged to -inf.
                            _ => f64::NEG_INFINITY,
                        })
                        .collect();
                    out.with_column(Series::new(DERIVED_OFFSET.into(), logs))?;
                    offset_col = Some(DERIVED_OFFSET.to_string());
                }
            }
        }

        Ok(Prepared {
            df: out,
            encoding,
            weight_col,
            offset_col,
        })
    }
}

fn missing_column(column: &str, df: &DataFrame) -> PolarsError {
    PolarsError::ColumnNotFound(
        format!(
            "Column '{}' is named by the plan but is not in the data. Columns present: {}",
            column,
            df.get_column_names()
                .iter()
                .map(|c| c.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .into(),
    )
}

/// Cast a categorical column to `Int32`, mapping strings to codes when needed.
fn encode_categorical(
    series: &Column,
    column: &str,
    encoding: Option<&Encoding>,
) -> Result<(Column, Option<Vec<(String, i32)>>), PolarsError> {
    match series.dtype() {
        DataType::String | DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let text = series.cast(&DataType::String)?;
            let text = text.str()?;

            let map: Vec<(String, i32)> = match encoding.and_then(|e| e.maps.get(column)) {
                Some(existing) => existing.clone(),
                None => {
                    // Codes follow sorted level text, so the same data always encodes
                    // the same way regardless of row order.
                    let mut levels: Vec<String> = text
                        .into_iter()
                        .flatten()
                        .map(str::to_string)
                        .collect::<HashSet<_>>()
                        .into_iter()
                        .collect();
                    levels.sort();
                    levels
                        .into_iter()
                        .enumerate()
                        .map(|(i, level)| (level, i as i32))
                        .collect()
                }
            };
            let lookup: HashMap<&str, i32> =
                map.iter().map(|(t, c)| (t.as_str(), *c)).collect();
            let codes: Vec<i32> = text
                .into_iter()
                .map(|v| match v {
                    Some(text) => *lookup.get(text).unwrap_or(&UNSEEN_CODE),
                    None => UNSEEN_CODE,
                })
                .collect();
            Ok((
                Series::new(column.into(), codes).into(),
                Some(map),
            ))
        }
        DataType::Int32 => Ok((series.clone(), None)),
        DataType::Int8
        | DataType::Int16
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Ok((series.cast(&DataType::Int32)?, None)),
        DataType::Boolean => Ok((series.cast(&DataType::Int32)?, None)),
        other => Err(PolarsError::ComputeError(
            format!(
                "Column '{}' has dtype {:?} and is used as a categorical factor. Use an \
                 integer, boolean, string or categorical column, or declare it as a \
                 banded numeric term instead.",
                column, other
            )
            .into(),
        )),
    }
}

// ---------------------------------------------------------------- resolution

/// What a plan decided for one term, once it had seen the data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTerm {
    pub name: String,
    /// `banded`, `categorical`, `variate` or `interaction`.
    pub kind: String,
    pub columns: Vec<String>,
    /// Rows in the table this term produced.
    pub rows: usize,
    /// Free parameters it spends once anchored: `rows - 1` for a step table, `degree`
    /// for a variate.
    pub parameters: usize,
    /// Band upper bounds, for a banded or variate term.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edges: Option<Vec<f64>>,
    /// The level anchored at zero, as it reads in the data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_level: Option<String>,
    /// The values a variate's polynomial is taken over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variate_values: Option<Vec<f64>>,
}

/// A plan turned into a model, with what it decided along the way.
pub struct BuiltPlan {
    pub model: RatingModel,
    pub table_names: Vec<String>,
    pub resolved: Vec<ResolvedTerm>,
    pub encoding: Encoding,
    pub weight_col: Option<String>,
    pub offset_col: Option<String>,
}

/// Inclusive upper bounds for a set of breaks, always ending unbounded.
fn band_edges(values: &[f64], breaks: &Breaks, column: &str) -> Result<Vec<f64>, PolarsError> {
    let mut edges: Vec<f64> = match breaks {
        Breaks::Explicit { edges } => edges.clone(),
        Breaks::Quantile { n } => {
            if *n < 1 {
                return Err(PolarsError::ComputeError(
                    format!("Quantile bands for '{}' need n of at least 1.", column).into(),
                ));
            }
            let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
            if sorted.is_empty() {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Column '{}' has no finite values, so quantile bands cannot be cut.",
                        column
                    )
                    .into(),
                ));
            }
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (1..*n)
                .map(|i| {
                    // Nearest-rank, so an edge is always a value the column takes.
                    let rank = (i as f64 / *n as f64 * sorted.len() as f64).ceil() as usize;
                    sorted[rank.clamp(1, sorted.len()) - 1]
                })
                .collect()
        }
        Breaks::EqualWidth { n } => {
            if *n < 1 {
                return Err(PolarsError::ComputeError(
                    format!("Equal-width bands for '{}' need n of at least 1.", column).into(),
                ));
            }
            let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
            if finite.is_empty() {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Column '{}' has no finite values, so equal-width bands cannot be cut.",
                        column
                    )
                    .into(),
                ));
            }
            let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (1..*n)
                .map(|i| lo + (hi - lo) * i as f64 / *n as f64)
                .collect()
        }
    };

    edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    edges.dedup();
    edges.retain(|e| e.is_finite());
    // Every table needs a final unbounded row: an observation matching no row would
    // contribute nothing and be dropped from that table's update.
    edges.push(f64::INFINITY);
    Ok(edges)
}

/// Midpoints for a variate's bands, with the open top band extrapolated.
fn default_variate_values(edges: &[f64]) -> Vec<f64> {
    let mut values = Vec::with_capacity(edges.len());
    let mut lower: Option<f64> = None;
    for (i, &edge) in edges.iter().enumerate() {
        if edge.is_finite() {
            let mid = match lower {
                Some(l) => (l + edge) / 2.0,
                // The first band is open below; take its upper bound as the value.
                None => edge,
            };
            values.push(mid);
            lower = Some(edge);
        } else {
            // The open-ended top band: carry the previous band's width past its edge.
            let width = match (i >= 2, lower) {
                (true, Some(l)) => (l - edges[i - 2]).abs().max(f64::EPSILON),
                _ => lower.map(|l| l.abs().max(1.0) * 0.5).unwrap_or(1.0),
            };
            values.push(lower.unwrap_or(0.0) + width);
        }
    }
    values
}

fn numeric_values(df: &DataFrame, column: &str) -> Result<Vec<f64>, PolarsError> {
    Ok(df
        .column(column)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect())
}

fn category_codes(df: &DataFrame, column: &str) -> Result<Vec<i32>, PolarsError> {
    Ok(df
        .column(column)?
        .i32()?
        .into_iter()
        .map(|v| v.unwrap_or(UNSEEN_CODE))
        .collect())
}

/// Distinct codes, ordered with the chosen base first.
fn ordered_levels(
    codes: &[i32],
    weights: &[f64],
    base: &Base,
    encoding: &Encoding,
    column: &str,
) -> Result<Vec<i32>, PolarsError> {
    let mut exposure: BTreeMap<i32, f64> = BTreeMap::new();
    for (i, code) in codes.iter().enumerate() {
        if *code == UNSEEN_CODE {
            continue;
        }
        *exposure.entry(*code).or_insert(0.0) += weights.get(i).copied().unwrap_or(1.0);
    }
    if exposure.is_empty() {
        return Err(PolarsError::ComputeError(
            format!("Column '{}' has no usable levels.", column).into(),
        ));
    }

    let mut levels: Vec<i32> = exposure.keys().copied().collect();
    let chosen = match base {
        Base::First => levels[0],
        Base::MostExposed => exposure
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(code, _)| *code)
            .unwrap_or(levels[0]),
        Base::Level { value } => {
            let code = encoding
                .code_for(column, value)
                .or_else(|| value.parse::<i32>().ok())
                .ok_or_else(|| {
                    PolarsError::ComputeError(
                        format!(
                            "Base level '{}' for column '{}' is not a level of that column. \
                             Levels present: {}",
                            value,
                            column,
                            describe_levels(&levels, encoding, column)
                        )
                        .into(),
                    )
                })?;
            if !exposure.contains_key(&code) {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Base level '{}' for column '{}' does not appear in the data. \
                         Levels present: {}",
                        value,
                        column,
                        describe_levels(&levels, encoding, column)
                    )
                    .into(),
                ));
            }
            code
        }
    };

    levels.retain(|c| *c != chosen);
    levels.insert(0, chosen);
    Ok(levels)
}

fn describe_levels(levels: &[i32], encoding: &Encoding, column: &str) -> String {
    levels
        .iter()
        .take(20)
        .map(|c| match encoding.label_for(column, *c) {
            Some(text) => text.to_string(),
            None => c.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

impl Plan {
    /// Turn the plan into a model, deciding bands and base levels from the data.
    ///
    /// `prepared` must come from [`Plan::prepare`].
    pub fn build(&self, prepared: &Prepared) -> Result<BuiltPlan, PolarsError> {
        if self.terms.is_empty() {
            return Err(PolarsError::ComputeError(
                "A plan needs at least one term. Add one with Plan::with(Term::...).".into(),
            ));
        }

        let df = &prepared.df;
        let weights: Vec<f64> = match &prepared.weight_col {
            Some(col) => df
                .column(col)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(0.0))
                .collect(),
            None => vec![1.0; df.height()],
        };

        // Table zero is always the intercept, created here rather than by the caller.
        let mut tables = vec![RatingTable::new(
            DataFrame::new(vec![Series::new("Rating_Factor".into(), vec![0.0]).into()])?,
            None,
        )];
        let mut names = vec!["intercept".to_string()];
        let mut resolved = vec![ResolvedTerm {
            name: "intercept".to_string(),
            kind: "intercept".to_string(),
            columns: Vec::new(),
            rows: 1,
            parameters: 1,
            edges: None,
            base_level: None,
            variate_values: None,
        }];

        for term in &self.terms {
            let (table, info) = self.build_term(term, df, &weights, &prepared.encoding)?;
            names.push(info.name.clone());
            resolved.push(info);
            tables.push(table);
        }

        let mut seen = HashSet::new();
        for name in &names {
            if !seen.insert(name.clone()) {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Two terms both produce a table named '{}'. Each column may carry \
                         at most one term.",
                        name
                    )
                    .into(),
                ));
            }
        }

        let model = RatingModel::new(
            tables,
            crate::rating_model::LinkFunction::from_objective(&self.family),
        );

        Ok(BuiltPlan {
            model,
            table_names: names,
            resolved,
            encoding: prepared.encoding.clone(),
            weight_col: prepared.weight_col.clone(),
            offset_col: prepared.offset_col.clone(),
        })
    }

    fn build_term(
        &self,
        term: &Term,
        df: &DataFrame,
        weights: &[f64],
        encoding: &Encoding,
    ) -> Result<(RatingTable, ResolvedTerm), PolarsError> {
        match term {
            Term::Banded { column, breaks } => {
                let values = numeric_values(df, column)?;
                let edges = band_edges(&values, breaks, column)?;
                let table = RatingTable::new(
                    DataFrame::new(vec![
                        Series::new(column.into(), edges.clone()).into(),
                        Series::new("Rating_Factor".into(), vec![0.0; edges.len()]).into(),
                    ])?,
                    None,
                );
                let rows = edges.len();
                Ok((
                    table,
                    ResolvedTerm {
                        name: column.clone(),
                        kind: "banded".to_string(),
                        columns: vec![column.clone()],
                        rows,
                        parameters: rows.saturating_sub(1),
                        edges: Some(edges),
                        base_level: Some("lowest band".to_string()),
                        variate_values: None,
                    },
                ))
            }

            Term::Categorical { column, base } => {
                let codes = category_codes(df, column)?;
                let levels = ordered_levels(&codes, weights, base, encoding, column)?;
                let table = RatingTable::new(
                    DataFrame::new(vec![
                        Series::new(column.into(), levels.clone()).into(),
                        Series::new("Rating_Factor".into(), vec![0.0; levels.len()]).into(),
                    ])?,
                    None,
                );
                let base_label = encoding
                    .label_for(column, levels[0])
                    .map(str::to_string)
                    .unwrap_or_else(|| levels[0].to_string());
                let rows = levels.len();
                Ok((
                    table,
                    ResolvedTerm {
                        name: column.clone(),
                        kind: "categorical".to_string(),
                        columns: vec![column.clone()],
                        rows,
                        parameters: rows.saturating_sub(1),
                        edges: None,
                        base_level: Some(base_label),
                        variate_values: None,
                    },
                ))
            }

            Term::Variate {
                column,
                breaks,
                values,
                degree,
            } => {
                let data = numeric_values(df, column)?;
                let edges = band_edges(&data, breaks, column)?;
                let variate_values = match values {
                    Some(v) => {
                        if v.len() != edges.len() {
                            return Err(PolarsError::ComputeError(
                                format!(
                                    "Variate '{}' was given {} values for {} bands. Supply one \
                                     value per band, or leave them out to take the midpoints. \
                                     The bands are {:?}.",
                                    column,
                                    v.len(),
                                    edges.len(),
                                    edges
                                )
                                .into(),
                            ));
                        }
                        v.clone()
                    }
                    None => default_variate_values(&edges),
                };
                let table = RatingTable::new(
                    DataFrame::new(vec![
                        Series::new(column.into(), edges.clone()).into(),
                        Series::new("Rating_Factor".into(), vec![0.0; edges.len()]).into(),
                    ])?,
                    None,
                )
                .as_polynomial_variate(variate_values.clone(), *degree)?;
                Ok((
                    table,
                    ResolvedTerm {
                        name: column.clone(),
                        kind: "variate".to_string(),
                        columns: vec![column.clone()],
                        rows: edges.len(),
                        parameters: *degree,
                        edges: Some(edges),
                        base_level: Some("lowest band".to_string()),
                        variate_values: Some(variate_values),
                    },
                ))
            }

            Term::Interaction { columns, breaks } => {
                if columns.len() < 2 {
                    return Err(PolarsError::ComputeError(
                        "An interaction needs at least two columns.".into(),
                    ));
                }
                if breaks.len() != columns.len() {
                    return Err(PolarsError::ComputeError(
                        format!(
                            "Interaction on {:?} was given {} break specifications for {} \
                             columns. Supply one per column, using None for a categorical.",
                            columns,
                            breaks.len(),
                            columns.len()
                        )
                        .into(),
                    ));
                }

                // Each axis's levels, ascending. Rows are generated in lexicographic
                // order over the axes, which is what makes first-match lookup correct
                // for a conjunction of upper bounds: for any fixed prefix the last
                // axis ascends, so the first row that satisfies every bound is the
                // tightest one.
                enum Axis {
                    Numeric(Vec<f64>),
                    Categorical(Vec<i32>),
                }
                let mut axes = Vec::with_capacity(columns.len());
                for (column, spec) in columns.iter().zip(breaks.iter()) {
                    match spec {
                        Some(breaks) => {
                            let values = numeric_values(df, column)?;
                            axes.push(Axis::Numeric(band_edges(&values, breaks, column)?));
                        }
                        None => {
                            let codes = category_codes(df, column)?;
                            axes.push(Axis::Categorical(ordered_levels(
                                &codes,
                                weights,
                                &Base::First,
                                encoding,
                                column,
                            )?));
                        }
                    }
                }

                let sizes: Vec<usize> = axes
                    .iter()
                    .map(|a| match a {
                        Axis::Numeric(v) => v.len(),
                        Axis::Categorical(v) => v.len(),
                    })
                    .collect();
                let total: usize = sizes.iter().product();
                if total == 0 {
                    return Err(PolarsError::ComputeError(
                        format!("Interaction on {:?} produced no rows.", columns).into(),
                    ));
                }

                // Odometer over the axes, last varying fastest.
                let mut numeric_cols: Vec<(String, Vec<f64>)> = Vec::new();
                let mut categorical_cols: Vec<(String, Vec<i32>)> = Vec::new();
                for (axis_index, axis) in axes.iter().enumerate() {
                    let repeat_inner: usize = sizes[axis_index + 1..].iter().product();
                    let repeat_outer: usize = sizes[..axis_index].iter().product();
                    match axis {
                        Axis::Numeric(levels) => {
                            let mut out = Vec::with_capacity(total);
                            for _ in 0..repeat_outer {
                                for level in levels {
                                    for _ in 0..repeat_inner {
                                        out.push(*level);
                                    }
                                }
                            }
                            numeric_cols.push((columns[axis_index].clone(), out));
                        }
                        Axis::Categorical(levels) => {
                            let mut out = Vec::with_capacity(total);
                            for _ in 0..repeat_outer {
                                for level in levels {
                                    for _ in 0..repeat_inner {
                                        out.push(*level);
                                    }
                                }
                            }
                            categorical_cols.push((columns[axis_index].clone(), out));
                        }
                    }
                }

                let mut series: Vec<Column> = Vec::new();
                for (name, values) in numeric_cols {
                    series.push(Series::new(name.as_str().into(), values).into());
                }
                for (name, values) in categorical_cols {
                    series.push(Series::new(name.as_str().into(), values).into());
                }
                series.push(Series::new("Rating_Factor".into(), vec![0.0; total]).into());

                Ok((
                    RatingTable::new(DataFrame::new(series)?, None),
                    ResolvedTerm {
                        name: term.name(),
                        kind: "interaction".to_string(),
                        columns: columns.clone(),
                        rows: total,
                        parameters: total.saturating_sub(1),
                        edges: None,
                        base_level: Some("first combination".to_string()),
                        variate_values: None,
                    },
                ))
            }
        }
    }
}

// ---------------------------------------------------------------- checking

/// Something worth saying about a plan before it is fitted.
#[derive(Debug, Clone)]
pub struct Issue {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    #[allow(dead_code)]
    pub column: Option<String>,
}

impl Issue {
    fn new(severity: Severity, code: &str, column: Option<&str>, message: String) -> Self {
        Self {
            severity,
            code: code.to_string(),
            message,
            column: column.map(str::to_string),
        }
    }
}

/// What a plan would do, and everything wrong with the data, before a fit is run.
#[derive(Debug, Clone)]
pub struct PlanCheck {
    /// What the plan decided, one entry per table including the intercept.
    pub resolved: Vec<ResolvedTerm>,
    /// Free parameters the model would spend.
    pub parameters: usize,
    /// Rows the model would carry across all tables.
    pub rows: usize,
    /// How strongly the tables share one direction, as
    /// [`crate::glm::collective_strength`]. Above about 10 the table solver needs
    /// hundreds of sweeps.
    pub table_conditioning: Option<f64>,
    /// Pairs of tables correlated above the near-alias threshold, as
    /// `(name_a, name_b, rho)`.
    pub correlated_pairs: Vec<(String, String, f64)>,
    /// Findings, most severe first.
    pub issues: Vec<Issue>,
}

impl PlanCheck {
    /// A check carrying no resolution, for a plan that could not be built.
    fn empty() -> Self {
        Self {
            resolved: Vec::new(),
            parameters: 0,
            rows: 0,
            table_conditioning: None,
            correlated_pairs: Vec::new(),
            issues: Vec::new(),
        }
    }

    fn blocked(issue: Issue) -> Self {
        Self {
            issues: vec![issue],
            ..Self::empty()
        }
    }

    /// True when nothing found would stop the plan being fitted usefully.
    pub fn is_fittable(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == Severity::High)
    }

    pub fn issues_at_least(&self, severity: Severity) -> Vec<&Issue> {
        self.issues
            .iter()
            .filter(|i| i.severity >= severity)
            .collect()
    }
}

impl Plan {
    /// Report what this plan would do and what is wrong with the data, without fitting.
    ///
    /// The point is to make one call answer every question a fit would otherwise
    /// surface one failed attempt at a time.
    pub fn check(&self, df: &DataFrame, target: &str) -> Result<PlanCheck, PolarsError> {
        let mut issues = Vec::new();

        // A check that throws on the first structural problem is no better than the
        // fit it replaces: the caller is back to discovering faults one failed attempt
        // at a time. So anything that would stop the plan being built is reported as a
        // finding, and `check` itself succeeds.
        let prepared = match self.prepare(df, None) {
            Ok(prepared) => prepared,
            Err(error) => {
                return Ok(PlanCheck::blocked(Issue::new(
                    Severity::High,
                    "plan_cannot_prepare",
                    None,
                    error.to_string(),
                )))
            }
        };

        // Structural faults are found before building, so the specific message
        // survives rather than being replaced by whatever the builder happened to
        // reject first.
        for term in &self.terms {
            if let Term::Variate {
                column,
                breaks,
                values,
                degree,
            } = term
            {
                let edges = match numeric_values(&prepared.df, column)
                    .and_then(|data| band_edges(&data, breaks, column))
                {
                    Ok(edges) => edges,
                    Err(error) => {
                        issues.push(Issue::new(
                            Severity::High,
                            "bands_cannot_be_cut",
                            Some(column),
                            error.to_string(),
                        ));
                        continue;
                    }
                };
                let variate_values = match values {
                    Some(v) => v.clone(),
                    None => default_variate_values(&edges),
                };
                let distinct = variate_values
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<HashSet<_>>()
                    .len();
                if *degree >= distinct {
                    issues.push(Issue::new(
                        Severity::High,
                        "variate_degree_too_high",
                        Some(column),
                        format!(
                            "Variate '{}' is degree {} over {} distinct band values. A degree \
                             at or above the number of distinct values is not identified — at \
                             {} the curve already passes through every band. Use degree {} or \
                             lower, or cut more bands.",
                            column,
                            degree,
                            distinct,
                            distinct.saturating_sub(1),
                            distinct.saturating_sub(1)
                        ),
                    ));
                }
            }
        }

        if issues.iter().any(|i| i.severity == Severity::High) {
            issues.sort_by(|a, b| b.severity.cmp(&a.severity));
            return Ok(PlanCheck {
                issues,
                ..PlanCheck::empty()
            });
        }

        let built = match self.build(&prepared) {
            Ok(built) => built,
            Err(error) => {
                issues.push(Issue::new(
                    Severity::High,
                    "plan_cannot_build",
                    None,
                    error.to_string(),
                ));
                issues.sort_by(|a, b| b.severity.cmp(&a.severity));
                return Ok(PlanCheck {
                    issues,
                    ..PlanCheck::empty()
                });
            }
        };

        // ---- target, weight and offset
        match prepared.df.column(target) {
            Err(_) => issues.push(Issue::new(
                Severity::High,
                "missing_target",
                Some(target),
                format!(
                    "Target column '{}' is not in the data. Columns present: {}",
                    target,
                    df.get_column_names()
                        .iter()
                        .map(|c| c.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )),
            Ok(series) => {
                if series.cast(&DataType::Float64).is_err() {
                    issues.push(Issue::new(
                        Severity::High,
                        "target_not_numeric",
                        Some(target),
                        format!(
                            "Target column '{}' has dtype {:?} and cannot be read as a number.",
                            target,
                            series.dtype()
                        ),
                    ));
                } else {
                    let values = series.cast(&DataType::Float64)?;
                    let ca = values.f64()?;
                    let bad = ca.into_iter().filter(|v| !matches!(v, Some(x) if x.is_finite())).count();
                    if bad > 0 {
                        issues.push(Issue::new(
                            Severity::High,
                            "target_has_nulls",
                            Some(target),
                            format!(
                                "Target column '{}' has {} null or non-finite values. Fitting \
                                 rejects them; drop or impute those rows first.",
                                target, bad
                            ),
                        ));
                    }
                    let negative = ca
                        .into_iter()
                        .flatten()
                        .filter(|v| *v < 0.0)
                        .count();
                    let needs_non_negative = matches!(
                        self.family.to_lowercase().as_str(),
                        "poisson" | "gamma" | "tweedie" | "binomial" | "binary"
                    );
                    if negative > 0 && needs_non_negative {
                        issues.push(Issue::new(
                            Severity::High,
                            "negative_target",
                            Some(target),
                            format!(
                                "Target column '{}' has {} negative values, which the {} family \
                                 cannot represent.",
                                target, negative, self.family
                            ),
                        ));
                    }
                }
            }
        }

        if let Some(exposure) = &self.exposure {
            if let Ok(series) = prepared.df.column(exposure) {
                if let Ok(ca) = series.f64() {
                    let non_positive = ca.into_iter().flatten().filter(|v| *v <= 0.0).count();
                    let null = ca.into_iter().filter(|v| v.is_none()).count();
                    if null > 0 {
                        issues.push(Issue::new(
                            Severity::High,
                            "exposure_has_nulls",
                            Some(exposure),
                            format!("Exposure column '{}' has {} null values.", exposure, null),
                        ));
                    }
                    if non_positive > 0 {
                        let role = self.resolved_exposure_role();
                        issues.push(Issue::new(
                            if role == ExposureRole::Offset {
                                Severity::High
                            } else {
                                Severity::Medium
                            },
                            "non_positive_exposure",
                            Some(exposure),
                            match role {
                                ExposureRole::Offset => format!(
                                    "Exposure column '{}' has {} values at or below zero. It \
                                     enters as log(exposure), which is undefined there, so \
                                     those rows must be dropped.",
                                    exposure, non_positive
                                ),
                                ExposureRole::Weight => format!(
                                    "Exposure column '{}' has {} values at or below zero. Rows \
                                     with no weight contribute nothing to the fit.",
                                    exposure, non_positive
                                ),
                            },
                        ));
                    }
                }
            }
        }

        // ---- per-term data quality
        for (term, info) in self.terms.iter().zip(built.resolved.iter().skip(1)) {
            for column in term.columns() {
                let series = match prepared.df.column(column) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let nulls = series.null_count();
                if nulls > 0 {
                    issues.push(Issue::new(
                        Severity::High,
                        "feature_has_nulls",
                        Some(column),
                        format!(
                            "Column '{}' has {} null values. An observation that fails to match \
                             a table is rejected by fitting, so fill or drop them first.",
                            column, nulls
                        ),
                    ));
                }
                if prepared.encoding.is_encoded(column) {
                    if let Ok(codes) = series.i32() {
                        let unseen = codes
                            .into_iter()
                            .flatten()
                            .filter(|c| *c == UNSEEN_CODE)
                            .count();
                        if unseen > 0 {
                            issues.push(Issue::new(
                                Severity::High,
                                "unencodable_levels",
                                Some(column),
                                format!(
                                    "Column '{}' has {} values that are not levels of the \
                                     encoding this plan was built with.",
                                    column, unseen
                                ),
                            ));
                        }
                    }
                }
                if let Ok(values) = series.f64() {
                    let non_finite = values
                        .into_iter()
                        .flatten()
                        .filter(|v| !v.is_finite())
                        .count();
                    if non_finite > 0 {
                        issues.push(Issue::new(
                            Severity::High,
                            "feature_not_finite",
                            Some(column),
                            format!(
                                "Column '{}' has {} infinite or NaN values, which cannot be \
                                 placed in a band.",
                                column, non_finite
                            ),
                        ));
                    }
                }
                if series.n_unique().unwrap_or(2) <= 1 {
                    issues.push(Issue::new(
                        Severity::Medium,
                        "constant_feature",
                        Some(column),
                        format!(
                            "Column '{}' takes a single value, so it carries no information and \
                             its factor will be aliased with the intercept.",
                            column
                        ),
                    ));
                }
            }

            if info.rows > WIDE_TABLE {
                issues.push(Issue::new(
                    Severity::Medium,
                    "wide_table",
                    None,
                    format!(
                        "Term '{}' produces {} rows. Every row is a free parameter unless the \
                         term is a variate, and a table this wide will be thinly estimated.",
                        info.name, info.rows
                    ),
                ));
            } else if info.kind == "categorical" && info.rows > WIDE_FACTOR {
                issues.push(Issue::new(
                    Severity::Low,
                    "many_levels",
                    Some(&info.columns[0]),
                    format!(
                        "Column '{}' has {} levels. Consider grouping the thin ones, or a \
                         penalty to shrink them toward the base level.",
                        info.columns[0], info.rows
                    ),
                ));
            }

        }

        // ---- thin levels, measured on the exposure the fit will use
        let weights: Vec<f64> = match &built.weight_col {
            Some(col) => prepared
                .df
                .column(col)?
                .f64()?
                .into_iter()
                .map(|v| v.unwrap_or(0.0))
                .collect(),
            None => vec![1.0; prepared.df.height()],
        };
        let total_weight: f64 = weights.iter().sum();
        let all_matches =
            crate::glm::matching::precompute_all_matches(&built.model, &prepared.df).ok();
        if total_weight > 0.0 {
            if let Some(matches) = all_matches.as_ref() {
                for (t, table) in built.model.tables.iter().enumerate().skip(1) {
                    let mut row_weight = vec![0.0; table.data.height()];
                    let mut unmatched = 0usize;
                    for (i, m) in matches[t].iter().enumerate() {
                        if *m == crate::glm::matching::NO_MATCH {
                            unmatched += 1;
                        } else {
                            row_weight[*m as usize] += weights[i];
                        }
                    }
                    if unmatched > 0 {
                        issues.push(Issue::new(
                            Severity::High,
                            "unmatched_observations",
                            None,
                            format!(
                                "{} rows do not match any row of table '{}'. Fitting rejects \
                                 them rather than dropping a term silently.",
                                unmatched, built.table_names[t]
                            ),
                        ));
                    }
                    let empty = row_weight.iter().filter(|w| **w <= 0.0).count();
                    if empty > 0 {
                        issues.push(Issue::new(
                            Severity::Medium,
                            "empty_levels",
                            None,
                            format!(
                                "{} of {} rows in table '{}' carry no exposure. They cannot be \
                                 estimated and will keep their starting factor.",
                                empty,
                                row_weight.len(),
                                built.table_names[t]
                            ),
                        ));
                    }
                    let thin = row_weight
                        .iter()
                        .filter(|w| **w > 0.0 && **w / total_weight < 0.001)
                        .count();
                    if thin > 0 {
                        issues.push(Issue::new(
                            Severity::Low,
                            "thin_levels",
                            None,
                            format!(
                                "{} rows in table '{}' hold under 0.1% of exposure each, so \
                                 their factors will be noisy.",
                                thin, built.table_names[t]
                            ),
                        ));
                    }
                }
            }
        }

        // ---- collinearity between tables, which is what makes a backfit crawl
        let mut correlated_pairs = Vec::new();
        let mut table_conditioning = None;
        if let Some(matches) = all_matches.as_ref() {
            // Offsets and variates never enter a pair: an offset carries no free
            // parameter, and a variate's rows are tied to a curve rather than free.
            let shapes: Vec<usize> = built
                .model
                .tables
                .iter()
                .map(|t| t.data.height())
                .collect();
            let eligible: Vec<bool> = built
                .model
                .tables
                .iter()
                .map(|t| !t.metadata.is_offset && t.variate_values().is_none())
                .collect();
            let pairs = crate::glm::table_correlations(matches, &weights, &shapes, &eligible);
            if !pairs.is_empty() {
                table_conditioning = Some(crate::glm::collective_strength(&pairs));
            }
            for pair in &pairs {
                if pair.correlation >= crate::glm::NEAR_ALIAS {
                    correlated_pairs.push((
                        built.table_names[pair.first].clone(),
                        built.table_names[pair.second].clone(),
                        pair.correlation,
                    ));
                }
            }
        }
        for (a, b, rho) in &correlated_pairs {
            let severity = if *rho > 0.999 {
                Severity::High
            } else {
                Severity::Medium
            };
            issues.push(Issue::new(
                severity,
                "near_aliased_tables",
                None,
                format!(
                    "Tables '{}' and '{}' are correlated at {:.4}. They describe nearly the \
                     same driver, so their factors are not separately identified{}",
                    a,
                    b,
                    rho,
                    if *rho > 0.999 {
                        " — drop one of them."
                    } else {
                        ". The fit solves such pairs jointly, but consider dropping one."
                    }
                ),
            ));
        }
        if let Some(conditioning) = table_conditioning {
            if conditioning > 10.0 {
                issues.push(Issue::new(
                    Severity::Medium,
                    "poorly_conditioned",
                    None,
                    format!(
                        "The tables share a common direction at strength {:.1}. Above about 10 \
                         the table solver needs hundreds of sweeps; use solver=\"auto\" so the \
                         global path can take it, or drop a redundant term.",
                        conditioning
                    ),
                ));
            }
        }

        let parameters: usize = built.resolved.iter().map(|r| r.parameters).sum();
        let rows: usize = built.resolved.iter().map(|r| r.rows).sum();

        if parameters >= prepared.df.height() {
            issues.push(Issue::new(
                Severity::High,
                "more_parameters_than_rows",
                None,
                format!(
                    "The plan spends {} parameters on {} rows of data. The fit is not \
                     identified.",
                    parameters,
                    prepared.df.height()
                ),
            ));
        }

        issues.sort_by(|a, b| b.severity.cmp(&a.severity));

        Ok(PlanCheck {
            resolved: built.resolved,
            parameters,
            rows,
            table_conditioning,
            correlated_pairs,
            issues,
        })
    }
}

// ---------------------------------------------------------------- fitting

/// A fitted plan: the model, what produced it, and everything needed to score,
/// validate and report on it.
pub struct FittedPlan {
    pub plan: Plan,
    pub model: RatingModel,
    pub diagnostics: GLMDiagnostics,
    pub table_names: Vec<String>,
    pub resolved: Vec<ResolvedTerm>,
    pub encoding: Encoding,
    pub target: String,
    pub weight_col: Option<String>,
    pub offset_col: Option<String>,
}

impl Plan {
    /// Prepare, build and fit in one call.
    pub fn fit(
        &self,
        df: &DataFrame,
        target: &str,
        mut options: GLMOptions,
    ) -> Result<FittedPlan, PolarsError> {
        let prepared = self.prepare(df, None)?;
        let built = self.build(&prepared)?;

        // The family lives on the plan, so the options cannot disagree with it.
        options.objective = self.family.clone();
        options.tweedie_power = self.tweedie_power;

        let (model, diagnostics) = fit_glm_with_diagnostics(
            &built.model,
            &prepared.df,
            target,
            built.weight_col.as_deref(),
            built.offset_col.as_deref(),
            options,
        )?;

        Ok(FittedPlan {
            plan: self.clone(),
            model,
            diagnostics,
            table_names: built.table_names,
            resolved: built.resolved,
            encoding: built.encoding,
            target: target.to_string(),
            weight_col: built.weight_col,
            offset_col: built.offset_col,
        })
    }
}

impl FittedPlan {
    /// Prepare scoring data with the encoding this model was fitted with.
    pub fn prepare(&self, df: &DataFrame) -> Result<Prepared, PolarsError> {
        self.plan.prepare(df, Some(&self.encoding))
    }

    /// Fitted means on the response scale.
    pub fn predict(&self, df: &DataFrame) -> Result<Series, PolarsError> {
        let prepared = self.prepare(df)?;
        self.model.predict(&prepared.df)
    }

    /// Validate against data, using the same weight and offset roles the fit used.
    pub fn validate(
        &self,
        df: &DataFrame,
        options: &ValidationOptions,
    ) -> Result<Validation, PolarsError> {
        let prepared = self.prepare(df)?;
        validate(
            &self.model,
            &prepared.df,
            &self.target,
            prepared.weight_col.as_deref(),
            prepared.offset_col.as_deref(),
            &self.plan.family,
            self.plan.tweedie_power,
            Some(&self.diagnostics),
            options,
        )
    }

    /// Rating tables with the inference joined on: `Coefficient`, `Standard_Error`,
    /// `Status`, and for log links `Relativity`. Categorical codes are rendered back
    /// into the level text they came from, as a `<column>_Level` column.
    pub fn rating_tables(&self) -> Result<Vec<DataFrame>, PolarsError> {
        let standard_errors = self
            .diagnostics
            .inference
            .as_ref()
            .map(|i| i.standard_errors.clone());
        let aliased = self
            .diagnostics
            .inference
            .as_ref()
            .map(|i| i.aliased_rows.clone())
            .unwrap_or_default();

        let mut out = Vec::with_capacity(self.model.tables.len());
        for (t, table) in self.model.tables.iter().enumerate() {
            let mut data = table.data.clone();
            let coefficients: Vec<f64> = data
                .column("Rating_Factor")?
                .f64()?
                .into_no_null_iter()
                .collect();
            let errors = standard_errors
                .as_ref()
                .and_then(|tables| tables.get(t))
                .cloned()
                .unwrap_or_else(|| vec![f64::NAN; coefficients.len()]);

            let status: Vec<&str> = (0..coefficients.len())
                .map(|r| {
                    if self.diagnostics.unfitted_rows.contains(&(t, r)) {
                        "no_data"
                    } else if aliased.contains(&(t, r)) {
                        "aliased"
                    } else if errors.get(r) == Some(&0.0) {
                        "reference"
                    } else {
                        "estimated"
                    }
                })
                .collect();

            // Put the level text back next to the codes, so the table reads.
            for column in table.data.get_column_names() {
                let name = column.to_string();
                if !self.encoding.is_encoded(&name) {
                    continue;
                }
                if let Ok(codes) = table.data.column(&name).and_then(|c| c.i32()) {
                    let labels: Vec<Option<String>> = codes
                        .into_iter()
                        .map(|c| {
                            c.and_then(|code| {
                                self.encoding.label_for(&name, code).map(str::to_string)
                            })
                        })
                        .collect();
                    data.with_column(Series::new(
                        format!("{}_Level", name).as_str().into(),
                        labels,
                    ))?;
                }
            }

            data.with_column(Series::new("Coefficient".into(), coefficients.clone()))?;
            data.with_column(Series::new("Standard_Error".into(), errors))?;
            data.with_column(Series::new("Status".into(), status))?;
            if self.model.get_link_function() == "log" {
                data.with_column(Series::new(
                    "Relativity".into(),
                    coefficients.iter().map(|v| v.exp()).collect::<Vec<f64>>(),
                ))?;
            }
            out.push(data);
        }
        Ok(out)
    }
}
