"""Fit, inspect and deliver models as rating tables.

Use Plan to specify a model, FittedModel to predict and review it, and Workbook to
save or edit its tables. from_booster converts supported LightGBM models with
optional parity evidence. Statistical review uses coefficient_intervals and
term_tests on supported original fits.

Data preparation, splits, study comparisons and selection loops belong in the
calling application; the examples demonstrate them with ordinary dataframe code.
"""
from .avenue_model import *  # noqa: F403
from . import avenue_model as _rust
from .tuning import TuningResult, resolve_lightgbm, supports_interaction_penalties, tune_lgbm
from .conversion import ConversionResult, from_booster
from .adapters import from_pandas
from .inference import CoefficientIntervals, coefficient_intervals, TermTests, term_tests

__all__ = [
    *(getattr(_rust, "__all__", None) or [n for n in dir(_rust) if not n.startswith("_")]),
    "TuningResult", "resolve_lightgbm", "supports_interaction_penalties", "tune_lgbm", "tuning",
    "ConversionResult", "from_booster", "from_pandas",
    "CoefficientIntervals", "coefficient_intervals", "TermTests", "term_tests",
]
