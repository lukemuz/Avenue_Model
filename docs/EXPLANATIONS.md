# Explain a quote and review a model change

`model.explain(quotes)` returns named `summary` and `contributions` Polars tables.
The summary retains original row position, linear predictor, link and final prediction.
The long contribution table names each term, its kind (intercept, table or exposure),
matched table row, coefficient and log-link multiplier. Table row positions are
zero-based within the current artifact and can be looked up in
`model.rating_tables_by_name()`.

Sum coefficients and apply the inverse link to reconstruct each response mean. Under
a log link, multiplying the base-rate and factor multipliers also reconstructs it.
For identity/logit links multipliers are null: effects add on the linear-predictor
scale. Exposure appears only when the model's declared mean uses an offset. Zero
exposure contributes negative infinity on that scale and a zero log-link multiplier.
Scoring errors remain strict; explanations never invent contributions for unmatched
rows. These are exact model mechanics, not causal attributions.

```python
from avenue_model import compare_changes

change = compare_changes(
    original, edited, quotes, unit="loss_per_exposure", weight="exposure",
    segments=["region"],
)
print(change.totals)
print(change.segments["region"])
print(change.policies.sort("weighted_change", descending=True).head(20))
print(change.contributions.filter(pl.col("coefficient_change") != 0))
```

Both artifacts score every row. They must share a link and recorded prediction
convention; the caller declares the common physical unit. Use exposure weights for
loss-per-exposure means, not for already exposure-adjusted totals. Policy and segment
changes reconcile to the weighted portfolio. Relative change is null for a zero old
mean. No rows are excluded.

Factor changes retain old/new table row positions and link-scale coefficients. Terms
are identified by `(kind, name)`; added/removed terms are explicit and contribute zero
on the absent side. Renaming a term is therefore reported as removal plus addition.
A nonlinear link means factor coefficient changes are not additive response-scale
loss-cost changes. For detailed structural review inspect both artifacts' tables.
The exhibit measures behavior on supplied quotes and does not certify unchanged
behavior elsewhere.

An edited workbook is a new scoring artifact. These exhibits do not assign old
standard errors to edited coefficients. Obtain fresh validation evidence for it.
The executable auto study demonstrates a known 5% factor edit, policy movements,
quote explanations and a separate edited-artifact validation report.

Large change reviews keep contribution tables columnar. The [controlled performance
study](CHANGE_REVIEW_PERFORMANCE.md) verifies unchanged million-row exhibits with
lower memory use and shorter runtime on a real six-table workload.
