// Core modules (always available)
pub mod analysis;
pub mod glm;
pub mod rating_model;
pub mod table_estimator;
pub mod validation;
pub mod tests;

// Python bindings (only when "python" feature is enabled)
#[cfg(feature = "python")]
use polars::prelude::*;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3_polars::PyDataFrame;
#[cfg(feature = "python")]
use rating_model::RatingModel;
#[cfg(feature = "python")]
use std::collections::HashMap;
#[cfg(feature = "python")]
use table_estimator::estimate_number_of_tables;

#[cfg(feature = "python")]
#[pyclass(name = "RatingModel")]
#[derive(Clone)]
struct PyRatingModel {
    inner: RatingModel,
    /// Statistical family used to construct the model. Keeping this beside the
    /// table-native core prevents the Python fitter from silently choosing a
    /// likelihood that disagrees with the model's link.
    family: String,
    table_names: Vec<String>,
}

#[cfg(feature = "python")]
fn canonical_family(value: &str) -> PyResult<String> {
    match value.to_lowercase().as_str() {
        "regression" | "gaussian" | "normal" => Ok("gaussian".to_string()),
        "binary" | "binomial" | "logistic" => Ok("binary".to_string()),
        "poisson" => Ok("poisson".to_string()),
        "gamma" => Ok("gamma".to_string()),
        "tweedie" => Ok("tweedie".to_string()),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown family '{}'. Expected 'gaussian', 'poisson', 'gamma', 'tweedie', or 'binary'.",
            other
        ))),
    }
}

#[cfg(feature = "python")]
fn default_table_names(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            if i == 0 {
                "intercept".to_string()
            } else {
                format!("table_{}", i)
            }
        })
        .collect()
}

#[cfg(feature = "python")]
/// Reports how much information each pair of rating tables shares.
///
/// Returns a list of (first_table, second_table, correlation) sorted worst first, where
/// the correlation is the first canonical correlation between the two tables' levels.
/// It is exactly the factor by which the shared direction survives each fitting sweep,
/// so a pair near 1.0 both fits slowly and cannot be interpreted level by level.
#[pyfunction]
#[pyo3(signature = (model, df, weight_col=None))]
fn table_correlations(
    model: &PyRatingModel,
    df: PyDataFrame,
    weight_col: Option<&str>,
) -> PyResult<Vec<(usize, usize, f64)>> {
    let df: DataFrame = df.into();
    let inner = &model.inner;

    let matches = glm::matching::precompute_all_matches(inner, &df)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let weights = match weight_col {
        Some(col) => {
            let ca = df
                .column(col)
                .and_then(|c| c.f64().cloned())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            (0..ca.len()).map(|i| ca.get(i).unwrap_or(0.0)).collect()
        }
        None => vec![1.0; df.height()],
    };

    let shapes: Vec<usize> = inner.tables.iter().map(|t| t.data.height()).collect();
    let eligible: Vec<bool> = inner
        .tables
        .iter()
        .map(|t| !t.metadata.is_offset && t.variate_values().is_none())
        .collect();

    Ok(
        glm::table_correlations(&matches, &weights, &shapes, &eligible)
            .into_iter()
            .map(|p| (p.first, p.second, p.correlation))
            .collect(),
    )
}

#[cfg(feature = "python")]
#[pyfunction]
fn estimate_num_tables(model_json: &str) -> PyResult<usize> {
    estimate_number_of_tables(model_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to estimate number of tables: {}",
            e
        ))
    })
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRatingModel {
    /// Create a new RatingModel from a collection of rating tables
    ///
    /// Args:
    ///     tables: List of polars DataFrames containing rating tables
    ///     objective: String indicating the model objective ("regression", "binary", etc)
    ///     feature_columns: Optional list of feature column names to use
    #[new]
    #[pyo3(signature = (tables, family, table_names=None, feature_columns=None, existing_row_number_col=None))]
    fn new(
        tables: Vec<PyDataFrame>,
        family: &str,
        table_names: Option<Vec<String>>,
        feature_columns: Option<Vec<String>>,
        existing_row_number_col: Option<&str>,
    ) -> PyResult<Self> {
        let family = canonical_family(family)?;
        let df_vec = tables
            .into_iter()
            .map(|pydf| pydf.0)
            .collect::<Vec<DataFrame>>();
        let names = table_names.unwrap_or_else(|| default_table_names(df_vec.len()));
        if names.len() != df_vec.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "table_names has {} entries but {} tables were supplied",
                names.len(),
                df_vec.len()
            )));
        }
        let mut seen = std::collections::HashSet::new();
        if names
            .iter()
            .any(|name| name.is_empty() || !seen.insert(name.clone()))
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "table_names must be non-empty and unique".to_string(),
            ));
        }

        RatingModel::from_dataframes(df_vec, &family, feature_columns, existing_row_number_col)
            .map(|model| PyRatingModel {
                inner: model,
                family,
                table_names: names,
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to create RatingModel: {}",
                    e
                ))
            })
    }

    /// Create a RatingModel from an LightGBM JSON model string
    ///
    /// Args:
    ///     model_json: JSON string representing the LightGBM model
    ///     consolidation_level: String indicating the consolidation level ("mean", "analysis")
    #[staticmethod]
    fn from_lgbm_json(model_json_str: &str, consolidation_level: &str) -> PyResult<Self> {
        let objective = serde_json::from_str::<serde_json::Value>(model_json_str)
            .ok()
            .and_then(|value| {
                value
                    .get("objective")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "gaussian".to_string());
        let family = canonical_family(objective.split_whitespace().next().unwrap_or("gaussian"))?;
        RatingModel::from_lgbm_json(&model_json_str, consolidation_level)
            .map(|model| {
                let table_names = default_table_names(model.tables.len());
                PyRatingModel {
                    inner: model,
                    family,
                    table_names,
                }
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to create model from JSON: {}",
                    e
                ))
            })
    }

    /// Get the link function used by the model
    fn get_link_function(&self) -> String {
        self.inner.get_link_function()
    }

    #[getter]
    fn family(&self) -> String {
        self.family.clone()
    }

    #[getter]
    fn table_names(&self) -> Vec<String> {
        self.table_names.clone()
    }

    /// Constrain one table's factors to a polynomial curve, so it costs the fit
    /// `degree` parameters instead of one per row.
    ///
    /// The table is still read as an ordinary step table — lookup and deployment are
    /// unchanged — but its fitted factors will all lie exactly on a degree-`degree`
    /// polynomial in `values`.
    ///
    /// Args:
    ///     table_index: Which table to constrain.
    ///     values: One number per row, in row order: what that row is worth on the
    ///         driver's scale. For an age table with bounds [20, 30, 40, 50, inf],
    ///         [20, 30, 40, 50, 65] is a natural choice — the last entry stands in
    ///         for the open-ended top band, which is why these are supplied rather
    ///         than taken from the table's own bounds column.
    ///     degree: Polynomial degree. 1 (the default) is a straight line and costs
    ///         one parameter; 2 bends once and costs two. Must be below the number
    ///         of distinct values, since at that point the curve already passes
    ///         through every row.
    ///
    /// Returns:
    ///     A new RatingModel; the original is unchanged.
    #[pyo3(signature = (table_index, values, degree=1))]
    fn as_variate(&self, table_index: usize, values: Vec<f64>, degree: usize) -> PyResult<Self> {
        if table_index >= self.inner.tables.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Table index {} is out of bounds (0-{})",
                table_index,
                self.inner.tables.len() - 1
            )));
        }
        if table_index == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Table 0 is the intercept and has a single row; there is no line to fit."
                    .to_string(),
            ));
        }

        let mut model = self.inner.clone();
        let table = std::mem::replace(
            &mut model.tables[table_index],
            rating_model::RatingTable::new(DataFrame::empty(), None),
        );
        model.tables[table_index] = table
            .as_polynomial_variate(values, degree)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyRatingModel {
            inner: model,
            family: self.family.clone(),
            table_names: self.table_names.clone(),
        })
    }

    /// The fitted slope of a linear variate table.
    ///
    /// None for a step table, and for a variate of degree above 1 — a curve has no
    /// single slope. Use variate_coefficients() there.
    fn variate_slope(&self, table_index: usize) -> PyResult<Option<f64>> {
        self.inner
            .tables
            .get(table_index)
            .map(|t| t.variate_slope())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Table index {} is out of bounds",
                    table_index
                ))
            })
    }

    /// The fitted polynomial coefficients [beta_1, ..., beta_degree] on the raw
    /// scale, so that factor[r] = constant + sum of beta_m * values[r] ** m.
    ///
    /// None if that table is not a variate. The constant is not returned: anchoring
    /// has moved it into the intercept, so it is not a property of this table.
    fn variate_coefficients(&self, table_index: usize) -> PyResult<Option<Vec<f64>>> {
        self.inner
            .tables
            .get(table_index)
            .map(|t| t.variate_coefficients())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Table index {} is out of bounds",
                    table_index
                ))
            })
    }

    /// The polynomial degree of a variate table, or None for a step table.
    fn variate_degree(&self, table_index: usize) -> PyResult<Option<usize>> {
        self.inner
            .tables
            .get(table_index)
            .map(|t| t.variate_degree())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Table index {} is out of bounds",
                    table_index
                ))
            })
    }

    /// Predict for a single set of features
    fn predict_one(&self, features: &Bound<'_, PyDict>) -> PyResult<f64> {
        let feature_map: HashMap<String, f64> = features.extract()?;
        Ok(self.inner.predict_one(&feature_map))
    }

    /// Predict for multiple rows in a DataFrame
    fn predict<'py>(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        let df: DataFrame = df.0;
        self.inner
            .predict(&df)
            .map(|predictions| {
                DataFrame::new(vec![predictions.into()])
                    .map(PyDataFrame)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Failed to create prediction DataFrame: {}",
                            e
                        ))
                    })
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Prediction failed: {}", e))
            })?
    }

    /// Predict on the linear-predictor scale, before applying the inverse link.
    fn predict_linear(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        let df: DataFrame = df.0;
        let values = self.inner.predict_linear(&df).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Prediction failed: {}", e))
        })?;
        DataFrame::new(vec![Series::new("linear_prediction".into(), values).into()])
            .map(PyDataFrame)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Predict a Poisson frequency rate. This is an explicit alias for predict()
    /// that guards against accidentally treating another family as frequency.
    fn predict_rate(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        if self.family != "poisson" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "predict_rate() is for Poisson frequency models, not family='{}'",
                self.family
            )));
        }
        self.predict(df)
    }

    /// Predict expected claim counts as fitted frequency rate times exposure.
    fn predict_expected(&self, df: PyDataFrame, exposure_col: &str) -> PyResult<PyDataFrame> {
        if self.family != "poisson" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "predict_expected() is for Poisson frequency models, not family='{}'",
                self.family
            )));
        }
        let df: DataFrame = df.0;
        let rates = self.inner.predict(&df).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Prediction failed: {}", e))
        })?;
        let exposure = df.column(exposure_col).and_then(|s| s.f64()).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Exposure column '{}' must exist and have dtype Float64",
                exposure_col
            ))
        })?;
        if exposure.null_count() > 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Exposure column '{}' contains null values",
                exposure_col
            )));
        }
        let expected: Vec<f64> = rates
            .f64()
            .unwrap()
            .into_no_null_iter()
            .zip(exposure.into_no_null_iter())
            .map(|(rate, exp)| rate * exp)
            .collect();
        DataFrame::new(vec![Series::new("expected".into(), expected).into()])
            .map(PyDataFrame)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Return each table's contribution on the linear-predictor scale.
    fn predict_components(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        let df: DataFrame = df.0;
        // Validate the complete model first so a missing match cannot become a
        // plausible-looking component value.
        self.inner.predict_linear(&df).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Prediction failed: {}", e))
        })?;
        let columns = self
            .inner
            .tables
            .iter()
            .zip(&self.table_names)
            .map(|(table, name)| Series::new(name.as_str().into(), table.predict_batch(&df)).into())
            .collect();
        DataFrame::new(columns)
            .map(PyDataFrame)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Consolidate the model's rating tables
    fn consolidate_tables(&self) -> PyResult<Self> {
        let inner = self.inner.consolidate_tables();
        let table_names = default_table_names(inner.tables.len());
        Ok(PyRatingModel {
            inner,
            family: self.family.clone(),
            table_names,
        })
    }

    /// Get the model's rating tables as DataFrames
    fn model_tables(&self) -> PyResult<Vec<PyDataFrame>> {
        Ok(self
            .inner
            .model_tables()
            .into_iter()
            .map(|df| PyDataFrame(df))
            .collect())
    }

    /// Return one rating table by its stable name.
    fn table(&self, name: &str) -> PyResult<PyDataFrame> {
        let index = self
            .table_names
            .iter()
            .position(|candidate| candidate == name)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Unknown table '{}'. Available tables: {}",
                    name,
                    self.table_names.join(", ")
                ))
            })?;
        Ok(PyDataFrame(self.inner.tables[index].data.clone()))
    }

    /// Combine two RatingModels
    fn __add__(&self, other: &PyRatingModel) -> PyResult<Self> {
        if self.family != other.family {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot combine {} and {} models",
                self.family, other.family
            )));
        }
        let family = self.family.clone();
        self.inner
            .clone()
            .combine(&other.inner)
            .map(|combined| {
                let table_names = default_table_names(combined.tables.len());
                PyRatingModel {
                    inner: combined,
                    family,
                    table_names,
                }
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to combine models: {}",
                    e
                ))
            })
    }

    fn round_rating_factors(&self, num_decimals: i32) -> PyResult<Self> {
        Ok(PyRatingModel {
            inner: self.inner.round_rating_factors(num_decimals),
            family: self.family.clone(),
            table_names: self.table_names.clone(),
        })
    }

    /// Perform one-way analysis on the rating model
    ///
    /// Args:
    ///     df: DataFrame containing the data to analyze
    ///     target_column: Column name to analyze (what we're averaging)
    ///     weight_column: Optional column name to use as weights
    #[pyo3(signature = (df, target_column, weight_column=None))]
    fn one_way_analysis<'py>(
        &self,
        df: PyDataFrame,
        target_column: &str,
        weight_column: Option<&str>,
    ) -> PyResult<Vec<PyDataFrame>> {
        let df: DataFrame = df.0;
        // ⭐ OPTIMIZED: Pass references instead of owned values
        self.inner
            .one_way_analysis(&df, target_column, weight_column)
            .map(|dfs| dfs.into_iter().map(|df| PyDataFrame(df)).collect())
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "One-way analysis failed: {}",
                    e
                ))
            })
    }

    /// Perform one-way analysis on a single rating table
    ///
    /// Args:
    ///     table_index: Index of the table in the model
    ///     df: DataFrame containing the data to analyze
    ///     target_column: Column name to analyze (what we're averaging)
    ///     weight_column: Optional column name to use as weights
    #[pyo3(signature = (table_index, df, target_column, weight_column=None))]
    fn one_way_analysis_table<'py>(
        &self,
        table_index: usize,
        df: PyDataFrame,
        target_column: &str,
        weight_column: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        if table_index >= self.inner.tables.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Table index {} is out of bounds (0-{})",
                table_index,
                self.inner.tables.len() - 1
            )));
        }

        let table = &self.inner.tables[table_index];
        let df: DataFrame = df.0;

        // ⭐ OPTIMIZED: Pass reference instead of owned value
        table
            .one_way_analysis_table(&df, target_column, weight_column)
            .map(|df| PyDataFrame(df))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "One-way analysis failed: {}",
                    e
                ))
            })
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "GLMOptions")]
#[derive(Clone)]
struct PyGLMOptions {
    inner: glm::GLMOptions,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGLMOptions {
    /// The statistical family is taken from RatingModel, so the likelihood and
    /// link have one authoritative source.
    ///
    /// Args:
    ///     max_iterations: Maximum coordinate-descent sweeps over the tables.
    ///     tolerance: Stop once the relative change in deviance falls below this.
    ///     verbose: Print deviance each sweep.
    ///     tweedie_power: Variance power, Tweedie only. 1 = Poisson, 2 = Gamma.
    ///     normalization: How to anchor the over-parameterised tables —
    ///         "base_level" (default) puts each table's first row at zero,
    ///         "weighted_mean" centres each table on its exposure-weighted mean,
    ///         "none" leaves factors wherever the fit put them.
    ///     compute_standard_errors: Compute standard errors and fit statistics
    ///         after converging. Default True. Turn off for models with thousands
    ///         of levels, where the p x p inversion becomes the dominant cost.
    ///     accelerate: Accelerate the sweep with SQUAREM extrapolation. Default
    ///         True. Costs three parameter vectors of memory and pays for itself
    ///         many times over when tables are correlated. Turn it off only to
    ///         reproduce the unaccelerated sequence exactly.
    ///     solve_aliased_pairs_jointly: Update near-aliased table pairs as one
    ///         block rather than one at a time. Default True. Costs a sampled pass
    ///         over the data to measure how much information each pair of tables
    ///         shares; worth it whenever a plan carries two tables describing the
    ///         same thing, which otherwise converges at a crawl.
    ///     alpha: Penalty strength. Default 0.0, which fits the unpenalised model.
    ///         Scaled to mean the same thing as glum's alpha, so the two are
    ///         directly comparable.
    ///
    ///         Each level is shrunk toward **its table's first row**, not toward
    ///         zero, which is the same thing one-hot dummy coding with a dropped
    ///         reference level does elsewhere. Two consequences: the intercept is
    ///         never penalised, and the choice of base level now changes the fit,
    ///         so a table anchored on a thin or extreme level will shrink toward a
    ///         bad reference. Variate tables are not penalised - a low-degree
    ///         polynomial is already the constraint a penalty would impose.
    ///
    ///         Requires normalization="base_level", the default. No standard
    ///         errors are reported once it is non-zero: the estimate is biased on
    ///         purpose, by about as much as the variance the penalty bought, so a
    ///         Wald interval centred on it would not have the coverage it appears
    ///         to claim. effective_parameters is still reported.
    ///     l1_ratio: How much of alpha is spent on the L1 term. 0.0 (default) is a
    ///         pure ridge, 1.0 a pure lasso, anything between an elastic net.
    ///
    ///         In the table solver a soft threshold replaces a division in the
    ///         existing per-level update. It also turns off the joint solve for
    ///         near-aliased table pairs, so correlated plans can need many sweeps.
    ///         The global solver uses coordinate descent on its Gram matrix instead.
    ///
    ///         No standard errors are reported under a penalty of either kind.
    ///         See GLMDiagnostics.standard_errors_note.
    ///     solver: "auto" (default) prefers global IRLS and falls back to the
    ///         low-memory table solver for unsupported or very wide models.
    ///         "global" requires the global path; "table" requires table descent.
    #[new]
    #[pyo3(signature = (
        max_iterations=None,
        tolerance=None,
        verbose=None,
        tweedie_power=None,
        normalization=None,
        compute_standard_errors=None,
        accelerate=None,
        solve_aliased_pairs_jointly=None,
        alpha=None,
        l1_ratio=None,
        solver=None,
    ))]
    fn new(
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
        verbose: Option<bool>,
        tweedie_power: Option<f64>,
        normalization: Option<&str>,
        compute_standard_errors: Option<bool>,
        accelerate: Option<bool>,
        solve_aliased_pairs_jointly: Option<bool>,
        alpha: Option<f64>,
        l1_ratio: Option<f64>,
        solver: Option<&str>,
    ) -> PyResult<Self> {
        let mut options = glm::GLMOptions::default();

        if let Some(max_iter) = max_iterations {
            options.max_iterations = max_iter;
        }
        if let Some(tol) = tolerance {
            options.tolerance = tol;
        }
        if let Some(v) = verbose {
            options.verbose = v;
        }
        if let Some(p) = tweedie_power {
            options.tweedie_power = p;
        }
        if let Some(n) = normalization {
            options.normalization = match n.to_lowercase().as_str() {
                "base_level" | "base" => glm::Normalization::BaseLevel,
                "weighted_mean" | "mean" => glm::Normalization::WeightedMean,
                "none" => glm::Normalization::None,
                other => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown normalization '{}'. Expected 'base_level', 'weighted_mean' or 'none'.",
                    other
                )))
                }
            };
        }

        if let Some(se) = compute_standard_errors {
            options.compute_standard_errors = se;
        }

        if let Some(a) = accelerate {
            options.accelerate = a;
        }

        if let Some(j) = solve_aliased_pairs_jointly {
            options.solve_aliased_pairs_jointly = j;
        }

        if let Some(a) = alpha {
            if !(a >= 0.0) || !a.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "alpha must be finite and not negative, got {}",
                    a
                )));
            }
            options.alpha = a;
        }

        if let Some(r) = l1_ratio {
            if !(0.0..=1.0).contains(&r) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "l1_ratio must be between 0 (ridge) and 1 (lasso), got {}",
                    r
                )));
            }
            options.l1_ratio = r;
        }

        if let Some(s) = solver {
            options.solver = match s.to_lowercase().as_str() {
                "auto" => glm::GLMSolver::Auto,
                "table" => glm::GLMSolver::Table,
                "global" => glm::GLMSolver::Global,
                other => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown solver '{}'. Expected 'auto', 'table', or 'global'.",
                        other
                    )))
                }
            };
        }

        Ok(PyGLMOptions { inner: options })
    }
}

/// Convergence and deviance information from a GLM fit.
#[cfg(feature = "python")]
#[pyclass(name = "GLMDiagnostics")]
#[derive(Clone)]
struct PyGLMDiagnostics {
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    converged: bool,
    /// Largest absolute score component at the final iterate, on the same scale as
    /// GLMOptions.tolerance. When converged is False, this says how far off the
    /// factors are.
    #[pyo3(get)]
    max_gradient: f64,
    /// Largest absolute score after each sweep. A sequence that falls steeply and
    /// then crawls is the signature of two near-aliased tables.
    #[pyo3(get)]
    gradient_history: Vec<f64>,
    #[pyo3(get)]
    deviance: f64,
    #[pyo3(get)]
    null_deviance: f64,
    #[pyo3(get)]
    deviance_history: Vec<f64>,
    /// Table rows that received no exposure and kept their starting factor,
    /// as (table_index, row_index) pairs.
    #[pyo3(get)]
    unfitted_rows: Vec<(usize, usize)>,
    /// How strongly the tables share a single common direction: 1.0 when they are
    /// orthogonal, up to the number of tables when they all carry the same
    /// information. This is what sets the sweep count on a correlated plan, and no
    /// pairwise correlation substitutes for it - a hundred tables pairwise-correlated
    /// at only 0.28 score 28.5 here and need about a thousand sweeps. Above roughly 10
    /// expect hundreds of sweeps; above 25, thousands. None when
    /// solve_aliased_pairs_jointly is off.
    #[pyo3(get)]
    table_conditioning: Option<f64>,
    /// Extrapolation steps the accelerator accepted. A large count next to a small
    /// `iterations` is SQUAREM earning its keep; zero means the fit converged
    /// before it was needed, or that acceleration is off.
    #[pyo3(get)]
    accelerated_steps: usize,
    /// Fraction of the null deviance explained by the fit.
    #[pyo3(get)]
    pseudo_r2: f64,

    // --- inference; None when compute_standard_errors is off ---
    /// Standard error of each table's rows, matching the model's table layout.
    /// A row that is the anchoring reference has a standard error of exactly 0.
    /// A row with no exposure, or one that is aliased, is NaN.
    #[pyo3(get)]
    standard_errors: Option<Vec<Vec<f64>>>,
    /// Why standard_errors is all NaN, when it is. None when they are real.
    /// Set on every penalised fit: a penalty biases the estimate on purpose, and
    /// a Wald interval centred on a shrunk coefficient does not have the coverage
    /// it looks like it has.
    #[pyo3(get)]
    standard_errors_note: Option<String>,
    /// Table rows whose effect cannot be separated from another parameter's, as
    /// (table_index, row_index) pairs. Usually two tables keyed on the same
    /// feature, or a completely separated level.
    #[pyo3(get)]
    aliased_rows: Option<Vec<(usize, usize)>>,
    /// 1 for Poisson and Binomial; Pearson chi-squared over residual degrees of
    /// freedom for Gaussian, Gamma and Tweedie.
    #[pyo3(get)]
    dispersion: Option<f64>,
    /// Free parameters actually estimated, i.e. the model's rank.
    #[pyo3(get)]
    n_parameters: Option<usize>,
    /// Effective parameters spent, which is what aic, bic and df_residual count.
    /// Equal to n_parameters without a penalty; below it with one, because a
    /// shrunk level costs less than a free one.
    #[pyo3(get)]
    effective_parameters: Option<f64>,
    #[pyo3(get)]
    df_residual: Option<f64>,
    #[pyo3(get)]
    pearson_chi2: Option<f64>,
    /// None for Tweedie, whose density has no closed form.
    #[pyo3(get)]
    log_likelihood: Option<f64>,
    #[pyo3(get)]
    aic: Option<f64>,
    #[pyo3(get)]
    bic: Option<f64>,
    /// Why standard errors are absent despite being requested. The fit is unaffected.
    #[pyo3(get)]
    inference_error: Option<String>,
    /// The fitted polynomial behind each variate table, one entry per variate table:
    /// (table_index, degree, raw-scale coefficients, standard errors, top-degree z).
    ///
    /// The standard errors are on the rescaled basis the fit uses. Because that basis
    /// is triangular, the top degree's z statistic does not depend on the scaling —
    /// it is the one that answers whether the curve needs to bend.
    #[pyo3(get)]
    variate_terms: Option<Vec<(usize, usize, Vec<f64>, Vec<f64>, Option<f64>)>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGLMDiagnostics {
    /// Wald z statistic for one table row: the factor divided by its standard error.
    ///
    /// Returns None for reference levels, aliased rows, and rows with no exposure.
    fn z_value(&self, table_index: usize, row_index: usize, factor: f64) -> Option<f64> {
        let se = *self
            .standard_errors
            .as_ref()?
            .get(table_index)?
            .get(row_index)?;
        if se > 0.0 && se.is_finite() {
            Some(factor / se)
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        let inference = match self.dispersion {
            Some(d) => format!(
                ", dispersion={:.6}, n_parameters={}, aic={}",
                d,
                self.n_parameters.unwrap_or(0),
                match self.aic {
                    Some(a) => format!("{:.4}", a),
                    None => "None".to_string(),
                }
            ),
            None => String::new(),
        };
        let conditioning = match self.table_conditioning {
            Some(c) => format!(", table_conditioning={:.1}", c),
            None => String::new(),
        };
        format!(
            "GLMDiagnostics(iterations={}, converged={}, max_gradient={:.2e}, deviance={:.6}, null_deviance={:.6}, pseudo_r2={:.4}{}{})",
            self.iterations, self.converged, self.max_gradient, self.deviance,
            self.null_deviance, self.pseudo_r2, conditioning, inference
        )
    }
}

#[cfg(feature = "python")]
impl From<glm::GLMDiagnostics> for PyGLMDiagnostics {
    fn from(d: glm::GLMDiagnostics) -> Self {
        let pseudo_r2 = d.pseudo_r2();
        let inf = d.inference;
        PyGLMDiagnostics {
            iterations: d.iterations,
            converged: d.converged,
            max_gradient: d.max_gradient,
            gradient_history: d.gradient_history,
            deviance: d.deviance,
            null_deviance: d.null_deviance,
            deviance_history: d.deviance_history,
            unfitted_rows: d.unfitted_rows,
            table_conditioning: d.table_conditioning,
            accelerated_steps: d.accelerated_steps,
            pseudo_r2,
            standard_errors: inf.as_ref().map(|i| i.standard_errors.clone()),
            standard_errors_note: inf.as_ref().and_then(|i| i.standard_errors_note.clone()),
            aliased_rows: inf.as_ref().map(|i| i.aliased_rows.clone()),
            dispersion: inf.as_ref().map(|i| i.dispersion),
            n_parameters: inf.as_ref().map(|i| i.n_parameters),
            effective_parameters: inf.as_ref().map(|i| i.effective_parameters),
            df_residual: inf.as_ref().map(|i| i.df_residual),
            pearson_chi2: inf.as_ref().map(|i| i.pearson_chi2),
            log_likelihood: inf.as_ref().and_then(|i| i.log_likelihood),
            aic: inf.as_ref().and_then(|i| i.aic),
            bic: inf.as_ref().and_then(|i| i.bic),
            inference_error: d.inference_error,
            variate_terms: inf.as_ref().map(|i| {
                i.variate_terms
                    .iter()
                    .map(|v| {
                        (
                            v.table_index,
                            v.degree,
                            v.coefficients.clone(),
                            v.standard_errors.clone(),
                            v.top_degree_z(),
                        )
                    })
                    .collect()
            }),
        }
    }
}

/// A cohesive fitted GLM: scoring model, diagnostics, and presentation-ready
/// rating tables stay together and share the same stable table names.
#[cfg(feature = "python")]
#[pyclass(name = "GLMResult")]
#[derive(Clone)]
struct PyGLMResult {
    model: PyRatingModel,
    diagnostics: PyGLMDiagnostics,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGLMResult {
    #[getter]
    fn model(&self) -> PyRatingModel {
        self.model.clone()
    }

    #[getter]
    fn diagnostics(&self) -> PyGLMDiagnostics {
        self.diagnostics.clone()
    }

    #[getter]
    fn converged(&self) -> bool {
        self.diagnostics.converged
    }

    /// Rating tables enriched with inferential columns and an explicit row status.
    /// Coefficient is on the linear-predictor scale; Relativity is exp(Coefficient)
    /// for log-link models and is omitted for identity and logit links.
    fn rating_tables(&self) -> PyResult<Vec<PyDataFrame>> {
        let aliased = self.diagnostics.aliased_rows.as_deref().unwrap_or(&[]);
        let mut result = Vec::with_capacity(self.model.inner.tables.len());
        for (table_index, table) in self.model.inner.tables.iter().enumerate() {
            let mut data = table.data.clone();
            let coefficients: Vec<f64> = data
                .column("Rating_Factor")
                .and_then(|s| s.f64())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                .into_no_null_iter()
                .collect();
            let standard_errors = self
                .diagnostics
                .standard_errors
                .as_ref()
                .and_then(|tables| tables.get(table_index))
                .cloned()
                .unwrap_or_else(|| vec![f64::NAN; coefficients.len()]);
            let status: Vec<&str> = (0..coefficients.len())
                .map(|row| {
                    if self.diagnostics.unfitted_rows.contains(&(table_index, row)) {
                        "no_data"
                    } else if aliased.contains(&(table_index, row)) {
                        "aliased"
                    } else if standard_errors.get(row) == Some(&0.0) {
                        "reference"
                    } else {
                        "estimated"
                    }
                })
                .collect();
            data.with_column(Series::new("Coefficient".into(), coefficients.clone()))
                .and_then(|df| {
                    df.with_column(Series::new("Standard_Error".into(), standard_errors))
                })
                .and_then(|df| df.with_column(Series::new("Status".into(), status)))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            if self.model.inner.get_link_function() == "log" {
                data.with_column(Series::new(
                    "Relativity".into(),
                    coefficients.iter().map(|v| v.exp()).collect::<Vec<_>>(),
                ))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            }
            result.push(PyDataFrame(data));
        }
        Ok(result)
    }

    fn predict(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        self.model.predict(df)
    }

    fn predict_linear(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        self.model.predict_linear(df)
    }

    fn predict_rate(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        self.model.predict_rate(df)
    }

    fn predict_expected(&self, df: PyDataFrame, exposure_col: &str) -> PyResult<PyDataFrame> {
        self.model.predict_expected(df, exposure_col)
    }

    fn predict_components(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        self.model.predict_components(df)
    }

    fn __repr__(&self) -> String {
        format!(
            "GLMResult(family='{}', tables={}, converged={}, iterations={}, deviance={:.6})",
            self.model.family,
            self.model.table_names.len(),
            self.diagnostics.converged,
            self.diagnostics.iterations,
            self.diagnostics.deviance,
        )
    }
}

/// Fit a GLM directly on the model's rating tables.
///
/// Args:
///     model: RatingModel supplying the table structure. Its factors are the
///         starting values; the structure is preserved.
///     df: Training data.
///     target_col: Column holding the response.
///     weight_col: Optional prior weight column (exposure, claim counts, ...).
///     offset_col: Optional column added to the linear predictor and held fixed.
///         For Poisson frequency this is normally log(exposure).
///     options: GLMOptions. Defaults to Gaussian if omitted.
///
/// Returns:
///     The fitted RatingModel.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (model, df, target_col, weight_col=None, offset_col=None, options=None))]
fn fit_glm(
    model: &PyRatingModel,
    df: PyDataFrame,
    target_col: &str,
    weight_col: Option<&str>,
    offset_col: Option<&str>,
    options: Option<PyGLMOptions>,
) -> PyResult<PyRatingModel> {
    let df: DataFrame = df.0;
    let mut glm_options = options.map(|o| o.inner).unwrap_or_default();
    glm_options.objective = model.family.clone();

    glm::fit_glm(
        &model.inner,
        &df,
        target_col,
        weight_col,
        offset_col,
        glm_options,
    )
    .map(|fitted_model| PyRatingModel {
        inner: fitted_model,
        family: model.family.clone(),
        table_names: model.table_names.clone(),
    })
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("GLM fitting failed: {}", e))
    })
}

/// Fit a model and return a cohesive result containing the fitted scoring model,
/// diagnostics, and presentation-ready rating tables.
///
/// Returns:
///     GLMResult
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (model, df, target_col, weight_col=None, offset_col=None, options=None))]
fn fit_glm_with_diagnostics(
    model: &PyRatingModel,
    df: PyDataFrame,
    target_col: &str,
    weight_col: Option<&str>,
    offset_col: Option<&str>,
    options: Option<PyGLMOptions>,
) -> PyResult<PyGLMResult> {
    let df: DataFrame = df.0;
    let mut glm_options = options.map(|o| o.inner).unwrap_or_default();
    glm_options.objective = model.family.clone();

    glm::fit_glm_with_diagnostics(
        &model.inner,
        &df,
        target_col,
        weight_col,
        offset_col,
        glm_options,
    )
    .map(|(fitted_model, diagnostics)| PyGLMResult {
        model: PyRatingModel {
            inner: fitted_model,
            family: model.family.clone(),
            table_names: model.table_names.clone(),
        },
        diagnostics: diagnostics.into(),
    })
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("GLM fitting failed: {}", e))
    })
}

#[cfg(feature = "python")]
#[pymodule]
fn avenue_model(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRatingModel>()?;
    m.add_class::<PyGLMOptions>()?;
    m.add_class::<PyGLMDiagnostics>()?;
    m.add_class::<PyGLMResult>()?;
    m.add_function(wrap_pyfunction!(estimate_num_tables, m)?)?;
    m.add_function(wrap_pyfunction!(fit_glm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_glm_with_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(table_correlations, m)?)?;
    Ok(())
}
