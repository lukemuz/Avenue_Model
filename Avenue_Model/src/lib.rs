// Core modules (always available)
pub mod license_handler;
pub mod rating_model;
pub mod table_estimator;
pub mod analysis;
pub mod tests;
pub mod glm;

// Python bindings (only when "python" feature is enabled)
#[cfg(feature = "python")]
use polars::frame::DataFrame;
#[cfg(feature = "python")]
use std::collections::HashMap;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::PyDataFrame;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use license_handler::internal_initialize_license;
#[cfg(feature = "python")]
use rating_model::RatingModel;
#[cfg(feature = "python")]
use table_estimator::estimate_number_of_tables;

#[cfg(feature = "python")]
#[pyfunction]
fn initialize_license(license_key: &str) -> PyResult<bool> {
    Ok(internal_initialize_license(license_key))
}

#[cfg(feature = "python")]
#[pyclass(name = "RatingModel")]
#[derive(Clone)]
struct PyRatingModel {
    inner: RatingModel,
}

#[cfg(feature = "python")]
#[pyfunction]
fn estimate_num_tables(model_json: &str) -> PyResult<usize> {
    estimate_number_of_tables(model_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to estimate number of tables: {}", e)
        ))
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
    fn new(
        tables: Vec<PyDataFrame>, 
        objective: &str, 
        feature_columns: Option<Vec<String>>,
        existing_row_number_col: Option<&str>
    ) -> PyResult<Self> {
        let df_vec = tables.into_iter()
            .map(|pydf| pydf.0)
            .collect::<Vec<DataFrame>>();
        
        RatingModel::from_dataframes(df_vec, objective, feature_columns, existing_row_number_col)
            .map(|model| PyRatingModel { inner: model })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to create RatingModel: {}", e)
            ))
    }

    /// Create a RatingModel from an LightGBM JSON model string
    /// 
    /// Args:
    ///     model_json: JSON string representing the LightGBM model
    ///     consolidation_level: String indicating the consolidation level ("mean", "analysis")
    #[staticmethod]
    fn from_lgbm_json(model_json_str: &str, consolidation_level: &str) -> PyResult<Self> {
        
        RatingModel::from_lgbm_json(&model_json_str, consolidation_level)
            .map(|model| PyRatingModel { inner: model })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to create model from JSON: {}", e)
            ))
    }

    /// Get the link function used by the model
    fn get_link_function(&self) -> String {
        self.inner.get_link_function()
    }

    /// Predict for a single set of features
    fn predict_one(&self, features: &Bound<'_, PyDict>) -> PyResult<f64> {
        let feature_map: HashMap<String, f64> = features.extract()?;
        Ok(self.inner.predict_one(&feature_map))
    }

    /// Predict for multiple rows in a DataFrame
    fn predict<'py>(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        let df: DataFrame = df.0;
        self.inner.predict(&df)
            .map(|predictions| {
                DataFrame::new(vec![predictions.into()])
                    .map(PyDataFrame)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Failed to create prediction DataFrame: {}", e)
                    ))
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Prediction failed: {}", e)
            ))?
    }

    /// Consolidate the model's rating tables
    fn consolidate_tables(&self) -> PyResult<Self> {
        Ok(PyRatingModel { 
            inner: self.inner.consolidate_tables() 
        })
    }

    /// Get the model's rating tables as DataFrames
    fn model_tables(&self) -> PyResult<Vec<PyDataFrame>> {
        Ok(self.inner.model_tables()
            .into_iter()
            .map(|df| PyDataFrame(df))
            .collect())
    }

    /// Combine two RatingModels
    fn __add__(&self, other: &PyRatingModel) -> PyResult<Self> {
        self.inner.clone().combine(&other.inner)
            .map(|combined| PyRatingModel { inner: combined })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to combine models: {}", e)
            ))
    }

    fn round_rating_factors(&self, num_decimals: i32) -> PyResult<Self> {
        Ok(PyRatingModel { 
            inner: self.inner.round_rating_factors(num_decimals) 
        })
    }

    /// Perform one-way analysis on the rating model
    /// 
    /// Args:
    ///     df: DataFrame containing the data to analyze
    ///     target_column: Column name to analyze (what we're averaging)
    ///     weight_column: Optional column name to use as weights
    fn one_way_analysis<'py>(&self, df: PyDataFrame, target_column: &str, weight_column: Option<&str>) -> PyResult<Vec<PyDataFrame>> {
        let df: DataFrame = df.0;
        // ⭐ OPTIMIZED: Pass references instead of owned values  
        self.inner.one_way_analysis(&df, target_column, weight_column)
            .map(|dfs| dfs.into_iter().map(|df| PyDataFrame(df)).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("One-way analysis failed: {}", e)
            ))
    }
    
    /// Perform one-way analysis on a single rating table
    /// 
    /// Args:
    ///     table_index: Index of the table in the model
    ///     df: DataFrame containing the data to analyze
    ///     target_column: Column name to analyze (what we're averaging)
    ///     weight_column: Optional column name to use as weights
    fn one_way_analysis_table<'py>(&self, table_index: usize, df: PyDataFrame, target_column: &str, weight_column: Option<&str>) -> PyResult<PyDataFrame> {
        if table_index >= self.inner.tables.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Table index {} is out of bounds (0-{})", table_index, self.inner.tables.len() - 1)
            ));
        }
        
        let table = &self.inner.tables[table_index];
        let df: DataFrame = df.0;
        
        // ⭐ OPTIMIZED: Pass reference instead of owned value
        table.one_way_analysis_table(&df, target_column, weight_column)
            .map(|df| PyDataFrame(df))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("One-way analysis failed: {}", e)
            ))
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
    #[new]
    fn new(
        objective: String, // Required parameter
        max_iterations: Option<usize>,
        tolerance: Option<f64>,
        verbose: Option<bool>,
        tweedie_power: Option<f64>,
    ) -> Self {
        let mut options = glm::GLMOptions::default();

        options.objective = objective;

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

        PyGLMOptions { inner: options }
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn fit_glm(
    model: &PyRatingModel,
    df: PyDataFrame,
    target_col: &str,
    weight_col: Option<&str>,
    options: Option<PyGLMOptions>,
) -> PyResult<PyRatingModel> {
    let df: DataFrame = df.0;
    let glm_options = options.map(|o| o.inner).unwrap_or_default();

    glm::fit_glm(&model.inner, &df, target_col, weight_col, glm_options)
        .map(|fitted_model| PyRatingModel { inner: fitted_model })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("GLM fitting failed: {}", e)
        ))
}

#[cfg(feature = "python")]
#[pymodule]
fn avenue_model(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRatingModel>()?;
    m.add_class::<PyGLMOptions>()?;
    m.add_function(wrap_pyfunction!(initialize_license, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_num_tables, m)?)?;
    m.add_function(wrap_pyfunction!(fit_glm, m)?)?;
    Ok(())
}

