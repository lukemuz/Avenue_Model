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

__all__ = [
    *(getattr(_rust, "__all__", None) or
      [n for n in dir(_rust) if not n.startswith("_")]),
    "TuningResult",
    "resolve_lightgbm",
    "supports_interaction_penalties",
    "tune_lgbm",
    "tuning",
]
