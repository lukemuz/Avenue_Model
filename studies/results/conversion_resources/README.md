# Final conversion resource limits

The original saved real booster has historical mean CV table count 3. The current
conversion contains 4 tables, 21,723 rows and a largest table of 21,080 rows.
Requesting `resource_limits={"tables": 3}` raises with the actual count and bound.
No constraint is inferred from the historical CV mean when none is requested.

The current conversion is measured anew; the older saved workbook used in the
large-table scoring benchmark has 19,181 rows. These are different artifact snapshots.
This resource check makes no new prediction-parity or speed claim.

Reproduce in the fork environment:

```python
import lightgbm as lgb
from avenue_model import from_booster
booster = lgb.Booster(model_file="evaluation/fork_results/fork_numeric/selected_booster.txt")
print(from_booster(booster).metadata["complexity"])
from_booster(booster, resource_limits={"tables": 3})  # expected ValueError: tables=4
```

All 147 Python tests pass, including exact-limit acceptance and saved metadata,
each exceeded bound, invalid requests before conversion, both consolidation modes,
and prediction preservation for an independently specified four-leaf tree. Its
explicit missing routes count toward the limits. Both stock and fork tutorials pass.
See [the conversion contract](../../../docs/CONVERSION.md) for scope and semantics.
