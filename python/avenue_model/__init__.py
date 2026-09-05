"""Avenue: rating tables you can file, fitted by a GLM or converted from LightGBM.

The compiled engine lives in `avenue_model.avenue_model` and is re-exported here. The
pure-Python additions are the parts that only make sense with a booster in hand, and
their dependencies are optional — see `avenue_model.tuning`.
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
