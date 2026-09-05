# Preparing and reviewing loss experience

```python
from avenue_model import prepare_pricing

prepared = prepare_pricing(
    data, exposure="earned_exposure", claims="claim_count", loss="trended_loss",
    predictors=["region", "age"], large_loss=100_000,
)
print(prepared.summary)
print(prepared.experience("region"))
frequency_data = prepared.frequency   # target avenue_frequency
severity_data = prepared.severity     # target avenue_severity
premium_data = prepared.pure_premium # target avenue_pure_premium
```

The helper preserves supplied source columns and appends `avenue_row` to identify
original row positions. Claims must be nonnegative integer counts. Exposure and loss
must be finite and nonnegative. Missing requested predictors, losses without claims,
and claim/loss activity without exposure are invalid. Invalid rows raise by default;
`invalid="exclude"` explicitly produces an exclusion audit instead.

Frequency and pure premium include valid positive-exposure records. Gamma severity
includes positive-exposure records with positive counts **and positive losses**.
Zero-loss claims remain in frequency and pure premium but are excluded from the
positive severity population. This makes the severity estimand conditional on positive
loss; assess whether a separate zero-payment probability model or a different severity
definition is needed before multiplying it by all-claim frequency. The helper does not
resolve that modeling decision automatically.

`audit` reports every original row's inclusion in each population, invalidity reasons,
and zero-exposure/zero-claim-loss/large-loss flags. `summary` reports input, included
and excluded row counts plus included exposure, claims and losses for each population.
`experience(factor)` uses the common frequency/pure-premium population and returns
rows, exposure, claims, loss, frequency, severity and pure premium. Its totals reconcile
to that population's summary. Severity ratios are null when a segment has no claims.

Large losses are flagged without capping. No source loss or exposure is adjusted.
Supply developed/trended losses, coverage/peril definitions, limits/deductibles and
other treatment rules explicitly before preparation and retain that provenance.

## Executable synthetic auto study

Run from the repository root:

```sh
python examples/auto_pricing_study.py --output /tmp/auto-study
```

The example generates a clearly labeled synthetic CSV and loads it, or accepts
`--data your.csv` with the documented column schema. It prepares populations, writes
experience/audit tables, uses a 2022 out-of-time holdout, checks convergence, writes
factor/uncertainty reviews, compares frequency-times-severity with a Tweedie mean
under a common loss, and saves/reloads editable workbooks to score quote predictors.
The population is simplified; it is a software workflow exercise, not empirical
insurance performance evidence. The original loss values are assumed already at the
selected development/trend level. The example is exercised in CI.

`model.rating_tables_by_name()` returns estimated factor tables keyed by name, so
review/export code does not need to align parallel lists manually. Existing
`rating_tables()` remains available.
