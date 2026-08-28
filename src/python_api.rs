//! Python bindings for plans, validation and reports.
//!
//! Kept separate from `lib.rs` because the surface here is the one a caller actually
//! drives: a plan states the model, `check` says what is wrong before a fit, and a
//! report says whether to believe the result. The lower-level `RatingModel` and
//! `fit_glm` bindings remain in `lib.rs` for callers building tables by hand.
//!
//! Structured results come back as plain dicts rather than opaque objects wherever
//! they are meant to be read, filtered or serialised — a finding a caller cannot
//! `json.dumps` is a finding that does not reach whoever needed it.

#![cfg(feature = "python")]

use crate::plan::{
    Base, Breaks, ExposureRole, FittedModel, GivenRole, Plan, PlanCheck, ResolvedTerm, Term,
};
use crate::report::{ModelReport, Verdict};
use crate::validation::{Severity, Validation, ValidationOptions};
use crate::workbook::{Scale, Workbook};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_polars::PyDataFrame;

fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(error.to_string())
}

/// Turn the three band spellings into one, refusing an ambiguous combination rather
/// than silently preferring one.
fn breaks_from(
    edges: Option<Vec<f64>>,
    quantile: Option<usize>,
    equal_width: Option<usize>,
) -> PyResult<Breaks> {
    match (edges, quantile, equal_width) {
        (Some(edges), None, None) => Ok(Breaks::explicit(edges)),
        (None, Some(n), None) => Ok(Breaks::quantile(n)),
        (None, None, Some(n)) => Ok(Breaks::equal_width(n)),
        (None, None, None) => Err(value_error(
            "Give exactly one of breaks=[...], quantile=n or equal_width=n.",
        )),
        _ => Err(value_error(
            "Give exactly one of breaks=[...], quantile=n or equal_width=n, not several.",
        )),
    }
}

fn scale_from(scale: Option<&str>) -> PyResult<Option<Scale>> {
    match scale {
        None => Ok(None),
        Some("relativity") => Ok(Some(Scale::Relativity)),
        Some("factor") => Ok(Some(Scale::Factor)),
        Some(other) => Err(value_error(format!(
            "scale must be 'relativity' or 'factor', got '{}'.",
            other
        ))),
    }
}

fn base_from(base: Option<&str>) -> PyResult<Base> {
    match base {
        None | Some("most_exposed") => Ok(Base::MostExposed),
        Some("first") => Ok(Base::First),
        Some(level) => Ok(Base::Level {
            value: level.to_string(),
        }),
    }
}

fn resolved_to_dict<'py>(py: Python<'py>, term: &ResolvedTerm) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &term.name)?;
    dict.set_item("kind", &term.kind)?;
    dict.set_item("columns", term.columns.clone())?;
    dict.set_item("rows", term.rows)?;
    dict.set_item("parameters", term.parameters)?;
    dict.set_item("edges", term.edges.clone())?;
    dict.set_item("base_level", term.base_level.clone())?;
    dict.set_item("variate_values", term.variate_values.clone())?;
    Ok(dict)
}

// ---------------------------------------------------------------- Plan

/// A model, stated rather than constructed.
///
/// Builder methods return a new plan, so a plan is never modified under a caller that
/// still holds it.
#[pyclass(name = "Plan")]
#[derive(Clone)]
pub struct PyPlan {
    pub(crate) inner: Plan,
}

#[pymethods]
impl PyPlan {
    /// A plan for an arbitrary family: one of `gaussian`, `poisson`, `gamma`,
    /// `tweedie`, `binomial`.
    #[new]
    #[pyo3(signature = (family, exposure=None, exposure_role=None, tweedie_power=None))]
    fn new(
        family: &str,
        exposure: Option<&str>,
        exposure_role: Option<&str>,
        tweedie_power: Option<f64>,
    ) -> PyResult<Self> {
        let mut plan = Plan::new(family);
        plan.exposure = exposure.map(str::to_string);
        plan.exposure_role = match exposure_role {
            None => None,
            Some("offset") => Some(ExposureRole::Offset),
            Some("weight") => Some(ExposureRole::Weight),
            Some(other) => {
                return Err(value_error(format!(
                    "exposure_role must be 'offset' or 'weight', got '{}'.",
                    other
                )))
            }
        };
        if let Some(power) = tweedie_power {
            plan.tweedie_power = power;
        }
        Ok(PyPlan { inner: plan })
    }

    /// Claim counts: Poisson with `log(exposure)` as an offset, so the fitted factors
    /// are rates and the target is a count.
    #[staticmethod]
    fn frequency(exposure: &str) -> Self {
        PyPlan {
            inner: Plan::frequency(exposure),
        }
    }

    /// Claim size: Gamma, weighted by claim count.
    #[staticmethod]
    fn severity(claim_count: &str) -> Self {
        PyPlan {
            inner: Plan::severity(claim_count),
        }
    }

    /// Loss per unit exposure: Tweedie, weighted by exposure.
    #[staticmethod]
    fn pure_premium(exposure: &str) -> Self {
        PyPlan {
            inner: Plan::pure_premium(exposure),
        }
    }

    /// A numeric driver cut into bands, each carrying its own free factor.
    ///
    /// Give exactly one of `breaks` (cut points), `quantile` (bands of roughly equal
    /// count) or `equal_width`. Cut points are inclusive upper bounds; a final
    /// unbounded band is always added, so no observation can fall outside the table.
    #[pyo3(signature = (column, breaks=None, quantile=None, equal_width=None))]
    fn banded(
        &self,
        column: &str,
        breaks: Option<Vec<f64>>,
        quantile: Option<usize>,
        equal_width: Option<usize>,
    ) -> PyResult<Self> {
        let breaks = breaks_from(breaks, quantile, equal_width)?;
        Ok(PyPlan {
            inner: self.inner.clone().with(Term::banded(column, breaks)),
        })
    }

    /// A categorical driver, one free factor per level.
    ///
    /// `base` is `most_exposed` (the default), `first`, or a level named as it appears
    /// in the data. The base level is anchored at zero, so every other level reads as
    /// a relativity against it.
    #[pyo3(signature = (column, base=None))]
    fn categorical(&self, column: &str, base: Option<&str>) -> PyResult<Self> {
        let base = base_from(base)?;
        Ok(PyPlan {
            inner: self
                .inner
                .clone()
                .with(Term::categorical_based_on(column, base)),
        })
    }

    /// A numeric driver whose band factors are tied to a polynomial, so the table
    /// costs `degree` parameters however many bands it has. It deploys as an ordinary
    /// step table — lookup is unchanged.
    #[pyo3(signature = (column, breaks=None, quantile=None, equal_width=None, degree=1, values=None))]
    fn variate(
        &self,
        column: &str,
        breaks: Option<Vec<f64>>,
        quantile: Option<usize>,
        equal_width: Option<usize>,
        degree: usize,
        values: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let breaks = breaks_from(breaks, quantile, equal_width)?;
        Ok(PyPlan {
            inner: self.inner.clone().with(Term::Variate {
                column: column.to_string(),
                breaks,
                values,
                degree,
            }),
        })
    }

    /// Several drivers crossed into one table.
    ///
    /// `breaks` is positional, one entry per column: a list of cut points bands that
    /// column, `None` treats it as categorical.
    #[pyo3(signature = (columns, breaks))]
    fn interaction(&self, columns: Vec<String>, breaks: Vec<Option<Vec<f64>>>) -> PyResult<Self> {
        if columns.len() != breaks.len() {
            return Err(value_error(format!(
                "interaction got {} columns and {} break specifications. Give one per \
                 column, using None for a categorical.",
                columns.len(),
                breaks.len()
            )));
        }
        let breaks = breaks
            .into_iter()
            .map(|b| b.map(Breaks::explicit))
            .collect();
        Ok(PyPlan {
            inner: self.inner.clone().with(Term::Interaction { columns, breaks }),
        })
    }

    /// A table supplied outright, whose levels and bands define the term and whose
    /// factors are re-estimated. The way to keep a plan's shape and refresh its numbers.
    fn given(&self, name: &str, table: PyDataFrame) -> PyResult<Self> {
        let table: DataFrame = table.into();
        Ok(PyPlan {
            inner: self
                .inner
                .clone()
                .with(Term::given(name, &table).map_err(value_error)?),
        })
    }

    /// A table supplied outright and held fixed: an existing rating plan that the new
    /// factors are fitted on top of. It contributes to every prediction and spends no
    /// parameters.
    fn offset(&self, name: &str, table: PyDataFrame) -> PyResult<Self> {
        let table: DataFrame = table.into();
        Ok(PyPlan {
            inner: self
                .inner
                .clone()
                .with(Term::offset(name, &table).map_err(value_error)?),
        })
    }

    /// Carry every table of a loaded model as a fixed offset.
    ///
    /// `prefix` namespaces the carried names — `"prior"` makes last year's `region`
    /// arrive as `prior.region`. It is required: the carried intercept would otherwise
    /// collide with this plan's own, and a report listing `region` twice says nothing
    /// about which is which.
    #[pyo3(signature = (model, prefix="prior"))]
    fn offset_model(&self, model: &PyFittedModel, prefix: &str) -> PyResult<Self> {
        let plan = self
            .inner
            .clone()
            .with_offset_model(
                &model.inner.model,
                &model.inner.table_names,
                prefix,
            )
            .map_err(value_error)?;
        // The carried tables hold category codes, so the plan must encode string
        // columns the same way or the same level could take a different code.
        Ok(PyPlan {
            inner: plan.with_encoding(model.inner.encoding.clone()),
        })
    }

    /// Encode string columns with these codes rather than deriving them from the data.
    /// Needed whenever the plan carries tables built elsewhere.
    fn with_encoding(&self, source: &PyFittedModel) -> Self {
        PyPlan {
            inner: self.inner.clone().with_encoding(source.inner.encoding.clone()),
        }
    }

    #[getter]
    fn family(&self) -> String {
        self.inner.family.clone()
    }

    #[getter]
    fn exposure(&self) -> Option<String> {
        self.inner.exposure.clone()
    }

    /// `offset` or `weight`, resolving the family default when it was not set.
    #[getter]
    fn exposure_role(&self) -> String {
        match self.inner.resolved_exposure_role() {
            ExposureRole::Offset => "offset".to_string(),
            ExposureRole::Weight => "weight".to_string(),
        }
    }

    #[getter]
    fn tweedie_power(&self) -> f64 {
        self.inner.tweedie_power
    }

    /// The names of the tables this plan will produce, intercept first.
    #[getter]
    fn term_names(&self) -> Vec<String> {
        let mut names = vec!["intercept".to_string()];
        names.extend(self.inner.terms.iter().map(|t| t.name()));
        names
    }

    /// The plan as JSON. This is the model's source code: save it, diff it, edit it,
    /// and load it back with `Plan.from_json`.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(PyPlan {
            inner: Plan::from_json(text).map_err(value_error)?,
        })
    }

    /// What this plan would do, and everything wrong with the data, without fitting.
    ///
    /// Reports rather than raises: a check that stopped at the first fault would leave
    /// the caller discovering problems one failed attempt at a time, which is the loop
    /// it exists to replace.
    fn check(&self, df: PyDataFrame, target: &str) -> PyResult<PyPlanCheck> {
        let df: DataFrame = df.into();
        Ok(PyPlanCheck {
            inner: self.inner.check(&df, target).map_err(value_error)?,
        })
    }

    /// Prepare, build and fit in one call.
    #[pyo3(signature = (df, target, options=None))]
    fn fit(
        &self,
        df: PyDataFrame,
        target: &str,
        options: Option<crate::PyGLMOptions>,
    ) -> PyResult<PyFittedModel> {
        let df: DataFrame = df.into();
        let options = options.map(|o| o.inner).unwrap_or_default();
        Ok(PyFittedModel {
            inner: self.inner.fit(&df, target, options).map_err(value_error)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Plan(family='{}', exposure={}, terms={})",
            self.inner.family,
            match &self.inner.exposure {
                Some(e) => format!("'{}' as {}", e, self.exposure_role()),
                None => "None".to_string(),
            },
            self.inner.terms.len()
        )
    }
}

// ---------------------------------------------------------------- PlanCheck

/// What a plan would do, and what is wrong with the data, before a fit.
#[pyclass(name = "PlanCheck")]
#[derive(Clone)]
pub struct PyPlanCheck {
    inner: PlanCheck,
}

#[pymethods]
impl PyPlanCheck {
    /// True when nothing found would stop the plan being fitted usefully.
    #[getter]
    fn is_fittable(&self) -> bool {
        self.inner.is_fittable()
    }

    /// What the plan decided, one dict per table: name, kind, columns, rows,
    /// parameters, edges, base_level, variate_values.
    ///
    /// This is where every default the plan applied is stated, so it can be relayed
    /// rather than assumed.
    #[getter]
    fn resolved<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for term in &self.inner.resolved {
            list.append(resolved_to_dict(py, term)?)?;
        }
        Ok(list)
    }

    /// Findings, most severe first. Each is a dict with `severity`, `code`, `message`
    /// and `column`. Branch on `code`; show `message` to a person unchanged.
    #[getter]
    fn issues<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for issue in &self.inner.issues {
            let dict = PyDict::new(py);
            dict.set_item("severity", issue.severity.as_str())?;
            dict.set_item("code", &issue.code)?;
            dict.set_item("message", &issue.message)?;
            dict.set_item("column", issue.column.clone())?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Free parameters the model would spend.
    #[getter]
    fn parameters(&self) -> usize {
        self.inner.parameters
    }

    /// Table rows the model would carry.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    /// How strongly the tables share one direction. Above about 10 the table solver
    /// needs hundreds of sweeps.
    #[getter]
    fn table_conditioning(&self) -> Option<f64> {
        self.inner.table_conditioning
    }

    /// Table pairs correlated above the near-alias threshold, as
    /// `(name_a, name_b, rho)`.
    #[getter]
    fn correlated_pairs(&self) -> Vec<(String, String, f64)> {
        self.inner.correlated_pairs.clone()
    }

    fn __repr__(&self) -> String {
        let high = self
            .inner
            .issues
            .iter()
            .filter(|i| i.severity == Severity::High)
            .count();
        format!(
            "PlanCheck(fittable={}, parameters={}, rows={}, issues={} ({} blocking))",
            self.inner.is_fittable(),
            self.inner.parameters,
            self.inner.rows,
            self.inner.issues.len(),
            high
        )
    }
}

// ---------------------------------------------------------------- Validation

/// The verdict on a fitted model, measured against data.
#[pyclass(name = "Validation")]
#[derive(Clone)]
pub struct PyValidation {
    inner: Validation,
}

#[pymethods]
impl PyValidation {
    /// True when nothing was found that should stop the model being used.
    #[getter]
    fn is_usable(&self) -> bool {
        self.inner.is_usable()
    }

    /// Findings, most severe first: `severity`, `code`, `message`, `rows`.
    #[getter]
    fn warnings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for warning in &self.inner.warnings {
            let dict = PyDict::new(py);
            dict.set_item("severity", warning.severity.as_str())?;
            dict.set_item("code", &warning.code)?;
            dict.set_item("message", &warning.message)?;
            dict.set_item("rows", warning.rows.clone())?;
            list.append(dict)?;
        }
        Ok(list)
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.n_rows
    }
    /// Rows that failed to match a table and were excluded. Non-zero means every
    /// figure here describes a subset of the data.
    #[getter]
    fn unmatched_rows(&self) -> usize {
        self.inner.unmatched_rows
    }
    #[getter]
    fn n_scored(&self) -> usize {
        self.inner.n_scored
    }
    #[getter]
    fn deviance(&self) -> f64 {
        self.inner.deviance
    }
    #[getter]
    fn null_deviance(&self) -> f64 {
        self.inner.null_deviance
    }
    /// Out of sample this can be negative, meaning the model does worse than the mean.
    #[getter]
    fn pseudo_r2(&self) -> f64 {
        self.inner.pseudo_r2
    }
    #[getter]
    fn total_actual(&self) -> f64 {
        self.inner.total_actual
    }
    #[getter]
    fn total_expected(&self) -> f64 {
        self.inner.total_expected
    }
    /// `total_actual / total_expected`. 1.0 is perfectly calibrated in aggregate.
    #[getter]
    fn ae_ratio(&self) -> f64 {
        self.inner.ae_ratio
    }
    /// Actual rate in the top bucket over the bottom one.
    #[getter]
    fn lift(&self) -> f64 {
        self.inner.lift
    }
    /// Weighted Gini on the predicted ordering. 0 is no discrimination.
    #[getter]
    fn gini(&self) -> f64 {
        self.inner.gini
    }

    /// Calibration and lift in one frame, on equal-exposure buckets ordered by
    /// prediction.
    #[getter]
    fn calibration(&self) -> PyDataFrame {
        PyDataFrame(self.inner.calibration.clone())
    }

    /// Actual versus expected per rating factor, one frame per table.
    #[getter]
    fn actual_vs_expected(&self) -> Vec<PyDataFrame> {
        self.inner
            .actual_vs_expected
            .iter()
            .cloned()
            .map(PyDataFrame)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Validation(scored={}/{}, ae_ratio={:.4}, gini={:.4}, pseudo_r2={:.4}, warnings={})",
            self.inner.n_scored,
            self.inner.n_rows,
            self.inner.ae_ratio,
            self.inner.gini,
            self.inner.pseudo_r2,
            self.inner.warnings.len()
        )
    }
}

// ---------------------------------------------------------------- FittedPlan

/// A fitted plan: the model, what produced it, and everything needed to score,
/// validate and report on it.
#[pyclass(name = "FittedModel")]
pub struct PyFittedModel {
    inner: FittedModel,
}

#[pymethods]
impl PyFittedModel {
    /// The plan that produced this, or `None` for a model that was loaded, converted
    /// or composed rather than fitted from a plan.
    #[getter]
    fn plan(&self) -> Option<PyPlan> {
        self.inner.plan.as_ref().map(|plan| PyPlan {
            inner: plan.clone(),
        })
    }

    /// Whether the fit converged. `None` when this model was not fitted here, so a
    /// caller cannot mistake "not fitted" for "did not converge".
    #[getter]
    fn converged(&self) -> Option<bool> {
        self.inner.converged()
    }

    /// True when this model was fitted here, rather than loaded or converted.
    #[getter]
    fn was_fitted(&self) -> bool {
        self.inner.was_fitted()
    }

    #[getter]
    fn family(&self) -> String {
        self.inner.family.clone()
    }

    /// The response column, when it is known.
    #[getter]
    fn target(&self) -> Option<String> {
        self.inner.target.clone()
    }

    /// The scoring model itself, for composition, consolidation and rounding.
    #[getter]
    fn rating_model(&self) -> crate::PyRatingModel {
        crate::PyRatingModel::wrap(
            self.inner.model.clone(),
            self.inner.family.clone(),
            self.inner.table_names.clone(),
        )
    }

    /// Non-blocking observations from loading this model out of a workbook.
    #[getter]
    fn notes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for note in &self.inner.notes {
            let dict = PyDict::new(py);
            dict.set_item("table", &note.table)?;
            dict.set_item("row", note.row)?;
            dict.set_item("code", &note.code)?;
            dict.set_item("message", &note.message)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Convert a LightGBM model into rating tables. Predictions match it exactly.
    ///
    /// `consolidation` is `"max"` for the minimal set of tables, `"analysis"` for one
    /// per tree node.
    #[staticmethod]
    #[pyo3(signature = (model_json, consolidation="max"))]
    fn from_lgbm_json(model_json: &str, consolidation: &str) -> PyResult<Self> {
        Ok(PyFittedModel {
            inner: FittedModel::from_lgbm_json(model_json, consolidation)
                .map_err(value_error)?,
        })
    }

    /// Say how this model's response was measured, so a converted or hand-built model
    /// can be validated on the same footing as a fitted one.
    ///
    /// `exposure_role` is `"offset"` (log exposure added to the linear predictor) or
    /// `"weight"`.
    #[pyo3(signature = (target, exposure=None, exposure_role=None))]
    fn with_response(
        &self,
        target: &str,
        exposure: Option<&str>,
        exposure_role: Option<&str>,
    ) -> PyResult<Self> {
        let role = match exposure_role {
            None => None,
            Some("offset") => Some(ExposureRole::Offset),
            Some("weight") => Some(ExposureRole::Weight),
            Some(other) => {
                return Err(value_error(format!(
                    "exposure_role must be 'offset' or 'weight', got '{}'.",
                    other
                )))
            }
        };
        Ok(PyFittedModel {
            inner: self.inner.clone().with_response(target, exposure, role),
        })
    }

    /// Add two models together.
    ///
    /// Under a log link the factors add, so the fitted means multiply — which makes a
    /// frequency model plus a severity model a pure premium model exactly. The result
    /// carries no fit statistics: it was composed, not fitted.
    fn combine(&self, other: &PyFittedModel) -> PyResult<Self> {
        Ok(PyFittedModel {
            inner: self.inner.combine(&other.inner).map_err(value_error)?,
        })
    }

    fn __add__(&self, other: &PyFittedModel) -> PyResult<Self> {
        self.combine(other)
    }

    #[getter]
    fn table_names(&self) -> Vec<String> {
        self.inner.table_names.clone()
    }

    /// What the plan decided, one dict per table.
    #[getter]
    fn resolved<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for term in &self.inner.resolved {
            list.append(resolved_to_dict(py, term)?)?;
        }
        Ok(list)
    }

    /// Fitted means on the response scale. Scoring goes through the same encoding the
    /// fit used, so a level maps to the code it was fitted with.
    fn predict(&self, df: PyDataFrame) -> PyResult<PyDataFrame> {
        let df: DataFrame = df.into();
        let predictions = self.inner.predict(&df).map_err(value_error)?;
        DataFrame::new(vec![predictions.into()])
            .map(PyDataFrame)
            .map_err(value_error)
    }

    /// Rating tables with `Coefficient`, `Standard_Error`, `Status` and, for log
    /// links, `Relativity`. Categorical codes carry their level text back as a
    /// `<column>_Level` column.
    fn rating_tables(&self) -> PyResult<Vec<PyDataFrame>> {
        Ok(self
            .inner
            .rating_tables()
            .map_err(value_error)?
            .into_iter()
            .map(PyDataFrame)
            .collect())
    }

    /// The fitted model as an editable, portable artifact.
    ///
    /// `scale` defaults to `"relativity"` under a log link, which is what a pricing
    /// actuary reads and edits, and `"factor"` otherwise.
    #[pyo3(signature = (scale=None))]
    fn to_workbook(&self, scale: Option<&str>) -> PyResult<PyWorkbook> {
        Ok(PyWorkbook {
            inner: self
                .inner
                .to_workbook(scale_from(scale)?)
                .map_err(value_error)?,
        })
    }

    /// Measure the model against data, in one call.
    #[pyo3(signature = (df, bins=None, bucket_tolerance=None, calibration_tolerance=None))]
    fn validate(
        &self,
        df: PyDataFrame,
        bins: Option<usize>,
        bucket_tolerance: Option<f64>,
        calibration_tolerance: Option<f64>,
    ) -> PyResult<PyValidation> {
        let df: DataFrame = df.into();
        let mut options = ValidationOptions::default();
        if let Some(bins) = bins {
            options.bins = bins;
        }
        if let Some(tolerance) = bucket_tolerance {
            options.bucket_tolerance = tolerance;
        }
        if let Some(tolerance) = calibration_tolerance {
            options.calibration_tolerance = tolerance;
        }
        Ok(PyValidation {
            inner: self.inner.validate(&df, &options).map_err(value_error)?,
        })
    }

    /// Assemble the full report, validating against `df` when it is given.
    ///
    /// The pre-fit check is carried automatically, so the plan's own findings arrive
    /// without being handed back.
    #[pyo3(signature = (df=None, bins=None))]
    fn report(
        &self,
        df: Option<PyDataFrame>,
        bins: Option<usize>,
    ) -> PyResult<PyModelReport> {
        let frame: Option<DataFrame> = df.map(|d| d.into());
        let mut options = ValidationOptions::default();
        if let Some(bins) = bins {
            options.bins = bins;
        }
        let report = self
            .inner
            .report(frame.as_ref(), &options)
            .map_err(value_error)?;
        Ok(PyModelReport { inner: report })
    }

    fn __repr__(&self) -> String {
        format!(
            "FittedModel(family='{}', target={}, tables={}, fitted={})",
            self.inner.family,
            match &self.inner.target {
                Some(target) => format!("'{}'", target),
                None => "None".to_string(),
            },
            self.inner.table_names.len(),
            self.inner.was_fitted()
        )
    }
}

// ---------------------------------------------------------------- ModelReport

/// Everything about one fitted model, in one place.
#[pyclass(name = "ModelReport")]
pub struct PyModelReport {
    inner: ModelReport,
}

#[pymethods]
impl PyModelReport {
    /// `usable`, `usable_with_caveats` or `not_usable`. The one field to branch on.
    #[getter]
    fn verdict(&self) -> String {
        self.inner.verdict.as_str().to_string()
    }

    /// One sentence, written to be relayed to a person unchanged. It always says what
    /// the numbers rest on, including when nothing was validated.
    #[getter]
    fn headline(&self) -> String {
        self.inner.headline.clone()
    }

    /// Everything worth saying, most severe first: `severity`, `code`, `message` and
    /// the `stage` it was found at (`plan`, `fit` or `validation`).
    #[getter]
    fn findings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for finding in &self.inner.findings {
            let dict = PyDict::new(py);
            dict.set_item("severity", finding.severity.as_str())?;
            dict.set_item("code", &finding.code)?;
            dict.set_item("message", &finding.message)?;
            dict.set_item("stage", &finding.stage)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// The whole report as Markdown, verdict and findings first.
    #[getter]
    fn markdown(&self) -> String {
        self.inner.to_markdown()
    }

    /// How the fit went: converged, iterations, deviance, pseudo_r2, aic, bic,
    /// dispersion, n_parameters, table_conditioning. Empty for a model that was loaded
    /// or converted rather than fitted here.
    #[getter]
    fn fit_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let Some(fit) = self.inner.fit_summary.as_ref() else {
            // Not fitted here, so there is nothing to report rather than zeroes that
            // would read as a fit that went badly.
            return Ok(dict);
        };
        dict.set_item("converged", fit.converged)?;
        dict.set_item("iterations", fit.iterations)?;
        dict.set_item("max_gradient", fit.max_gradient)?;
        dict.set_item("deviance", fit.deviance)?;
        dict.set_item("null_deviance", fit.null_deviance)?;
        dict.set_item("pseudo_r2", fit.pseudo_r2)?;
        dict.set_item("n_parameters", fit.n_parameters)?;
        dict.set_item("dispersion", fit.dispersion)?;
        dict.set_item("aic", fit.aic)?;
        dict.set_item("bic", fit.bic)?;
        dict.set_item("table_conditioning", fit.table_conditioning)?;
        Ok(dict)
    }

    #[getter]
    fn validation(&self) -> Option<PyValidation> {
        self.inner
            .validation
            .as_ref()
            .map(|v| PyValidation { inner: v.clone() })
    }

    /// A method, matching `FittedModel.rating_tables()`. One name should not change
    /// shape between classes.
    fn rating_tables(&self) -> Vec<PyDataFrame> {
        self.inner
            .rating_tables
            .iter()
            .cloned()
            .map(PyDataFrame)
            .collect()
    }

    #[getter]
    fn table_names(&self) -> Vec<String> {
        self.inner.table_names.clone()
    }

    #[getter]
    fn resolved<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for term in &self.inner.resolved {
            list.append(resolved_to_dict(py, term)?)?;
        }
        Ok(list)
    }

    /// The plan that produced this, as JSON. Load it with `Plan.from_json` to
    /// reproduce the model exactly.
    #[getter]
    fn plan_json(&self) -> String {
        self.inner.plan_json.clone()
    }

    /// A short digest of the plan, to tell two reports apart.
    #[getter]
    fn fingerprint(&self) -> String {
        self.inner.fingerprint.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelReport(verdict='{}', plan={}, findings={})",
            self.inner.verdict.as_str(),
            self.inner.fingerprint,
            self.inner.findings.len()
        )
    }
}

// ---------------------------------------------------------------- workbook

/// A model as a file you can open, edit and load back.
#[pyclass(name = "Workbook")]
#[derive(Clone)]
pub struct PyWorkbook {
    inner: Workbook,
}

#[pymethods]
impl PyWorkbook {
    /// Save as one self-contained JSON document.
    fn save_json(&self, path: &str) -> PyResult<()> {
        self.inner.save_json(path).map_err(value_error)
    }

    /// Save as a directory of CSVs plus `manifest.json` — the form to hand someone who
    /// will edit it in a spreadsheet.
    ///
    /// The manifest is not optional decoration: it carries the family, the scale, which
    /// tables are offsets, which rows are locked, the variates and the category codes.
    /// A directory of CSVs without it is not a model.
    fn save_csv_dir(&self, path: &str) -> PyResult<()> {
        self.inner.save_csv_dir(path).map_err(value_error)
    }

    #[staticmethod]
    fn load_json(path: &str) -> PyResult<Self> {
        Ok(PyWorkbook {
            inner: Workbook::load_json(path).map_err(value_error)?,
        })
    }

    #[staticmethod]
    fn load_csv_dir(path: &str) -> PyResult<Self> {
        Ok(PyWorkbook {
            inner: Workbook::load_csv_dir(path).map_err(value_error)?,
        })
    }

    /// Turn the workbook back into a model, checking every table first.
    ///
    /// The result is an ordinary `FittedModel`, so a loaded model scores, validates,
    /// reports and saves exactly like one just fitted.
    ///
    /// Raises if the structure is unusable, listing every fault at once rather than
    /// stopping at the first — a hand-edited file should be repairable in one pass.
    fn to_model(&self) -> PyResult<PyFittedModel> {
        Ok(PyFittedModel {
            inner: self.inner.to_model().map_err(value_error)?,
        })
    }

    /// The tables as they are written in the file, one frame each.
    #[getter]
    fn tables(&self) -> Vec<PyDataFrame> {
        self.inner.tables.iter().cloned().map(PyDataFrame).collect()
    }

    #[getter]
    fn table_names(&self) -> Vec<String> {
        self.inner
            .manifest
            .tables
            .iter()
            .map(|t| t.name.clone())
            .collect()
    }

    /// `"relativity"` or `"factor"` — which scale the factor column is on.
    #[getter]
    fn scale(&self) -> String {
        match self.inner.manifest.scale {
            Scale::Relativity => "relativity".to_string(),
            Scale::Factor => "factor".to_string(),
        }
    }

    #[getter]
    fn family(&self) -> String {
        self.inner.manifest.family.clone()
    }

    #[getter]
    fn link(&self) -> String {
        self.inner.manifest.link.clone()
    }

    /// The version of Avenue that wrote this file.
    #[getter]
    fn avenue_version(&self) -> String {
        self.inner.manifest.avenue_version.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Workbook(family='{}', tables={}, scale='{}')",
            self.inner.manifest.family,
            self.inner.tables.len(),
            self.scale()
        )
    }
}

// `GivenRole` is reached through Term::given and Term::offset above.
const _: Option<GivenRole> = None;

/// Register everything in this module on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPlan>()?;
    m.add_class::<PyPlanCheck>()?;
    m.add_class::<PyValidation>()?;
    m.add_class::<PyFittedModel>()?;
    m.add_class::<PyModelReport>()?;
    m.add_class::<PyWorkbook>()?;
    Ok(())
}

// Silences an unused-import warning when the crate is built without the classes being
// referenced elsewhere; `Verdict` is used through `as_str` above.
const _: Option<Verdict> = None;
