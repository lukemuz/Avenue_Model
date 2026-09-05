# Main effects plus a two-way interaction

```python
from avenue_model import Plan

plan = (
    Plan.frequency("exposure")
    .categorical("territory")
    .banded("driver_age", breaks=[25., 50.])
    .interaction(["territory", "driver_age"], [None, [25., 50.]])
)
model = plan.fit(training, "frequency")
print(model.resolved)
print(model.rating_tables_by_name())
```

When both corresponding main effects are present, Avenue recognizes a compatible
two-way categorical/banded interaction and uses identifiable treatment contrasts.
Interaction cells touching either main-effect reference level are fixed at zero.
Only the remaining interior cells are estimated. With `a` and `b` levels the interaction
therefore has `(a-1)*(b-1)` free parameters in a fully supported design. Reference
constraints are visible as locked table rows and in `resolved` as
`kind="interaction_contrast"`. They survive workbook export/reload.

Categorical references follow their main-effect selections, including most-exposed
levels; banded references are the first band. Terms may appear in any order. Numeric
axes must use the same break specification as their main effect so their learned bands
agree within each training fold. Main effects retain their usual interpretation at
the other factor's reference; the interaction is the additional joint effect on the
link scale.

The table solver supports the fixed reference rows. `solver="auto"` selects that
path; explicitly forcing the global solver still rejects partially locked tables.
Inference builds the actual free-parameter design and excludes all fixed reference
cells, rather than treating the resulting full cell table as unidentified. Independent
Poisson tests cover fitted means, estimable interaction contrasts and standard errors.
Sparse/unsupported cells can still be nonestimable and require review.

An interaction without both matching main effects retains the existing full-cell
representation. This increment does not implement arbitrary higher-order hierarchical
contrasts, automatic interaction selection, robust covariance or post-selection
uncertainty. Use the validation/complexity evidence to justify additional structure.
