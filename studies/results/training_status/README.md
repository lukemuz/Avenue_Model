# Learner-appropriate comparison status

Both the stock and fork CPU booster tutorials completed tuning, selected-schedule
training, conversion/reload parity, comparison and edit review. The compared boosters
have `training_status=completed`, `converged=null` and `eligible=true`. In both runs,
the baseline GLM still has lower holdout Poisson loss (0.808711 versus 0.811906).
Eligibility supplies no claim of predictive improvement.

All 132 Python tests passed. Focused comparison tests establish that explicit completed
training can make a candidate eligible without fabricating convergence; model-reported
nonconvergence, caller-reported nonconvergence, explicit training failure and scoring
failure cannot become a recommendation through that declaration. Unknown predictions
remain ineligible. A lower-loss completed candidate can be selected under the unchanged
common-loss rule.

Run `examples/booster_pricing_study.py --output <new-directory>` in the stock/fork
environment to reproduce. Full artifacts remain in `/tmp/avenue-booster-status-stock`
and `/tmp/avenue-booster-status-fork`. The manifest records their hashes, the parent
commit and modified source hashes. These are editable-source tutorial checks, not
fresh wheel acceptance or a new large real-data booster study.
