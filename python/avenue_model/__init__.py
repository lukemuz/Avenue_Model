"""Avenue: rating tables you can file, fitted by a GLM or converted from LightGBM.

Start with `Plan` to specify and fit a model, then use `FittedModel` for prediction,
factor review and editable `Workbook` export. `prepare_pricing`, `SplitSpec` and
`compare_models` support the surrounding pricing study. `frequency_severity` and
`sum_loss_costs` preserve explicit component units and scoring behavior.

`coefficient_intervals` and `term_tests` provide supported unpenalized inference.
`bootstrap_stability` supplies descriptive refit bands, while `poisson_credibility`
supplies posterior group relativities conditional on a fixed baseline and prior.
These methods have different statistical interpretations.

The compiled engine is re-exported from `avenue_model.avenue_model`. Optional extras
provide pandas adaptation, LightGBM tuning/conversion and credibility quantiles;
their dependencies are checked when the corresponding operation is called.
"""

from .avenue_model import *  # noqa: F403
from .avenue_model import __doc__ as _rust_doc  # noqa: F401

from . import avenue_model as _rust

# `tune_lgbm` needs lightgbm and optuna, which are an optional extra. Importing the
# names here rather than the module keeps `import avenue_model` working without them —
# the ImportError is raised when the function is called, with an actionable message.
from .tuning import (  # noqa: F401
    TuningResult,
    resolve_lightgbm,
    supports_interaction_penalties,
    tune_lgbm,
)

from .conversion import ConversionResult, from_booster
from .splitting import Fold, FoldFit, SplitSpec
from .comparison import Candidate, Comparison, compare_models
from .preparation import PreparedPricing, prepare_pricing
from .changes import ModelChange, compare_changes
from .adapters import from_pandas
from .composition import ComposedModel, frequency_severity, sum_loss_costs

__all__ = [
    "ComposedModel", "frequency_severity", "sum_loss_costs",
    "from_pandas",
    "ModelChange", "compare_changes",
    "Candidate", "Comparison", "compare_models", "PreparedPricing", "prepare_pricing",
    "ConversionResult", "from_booster", "Fold", "FoldFit", "SplitSpec",
    *(getattr(_rust, "__all__", None) or
      [n for n in dir(_rust) if not n.startswith("_")]),
    "TuningResult",
    "resolve_lightgbm",
    "supports_interaction_penalties",
    "tune_lgbm",
    "tuning",
]

from .bundle import ModelBundle, load_bundle, save_bundle
__all__ += ['ModelBundle', 'load_bundle', 'save_bundle']
from .selection import GLMSelection, GLMTrial, select_glm
__all__ += ['GLMSelection', 'GLMTrial', 'select_glm']
from .inference import CoefficientIntervals, coefficient_intervals, TermTests, term_tests
__all__ += ['CoefficientIntervals', 'coefficient_intervals', 'TermTests', 'term_tests']
from .credibility import CredibilityResult, poisson_credibility
__all__ += ['CredibilityResult', 'poisson_credibility']
from .stability import BootstrapStability, bootstrap_stability
__all__ += ['BootstrapStability', 'bootstrap_stability']
