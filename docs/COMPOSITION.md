# Named loss-cost composition

```python
from avenue_model import ComposedModel, frequency_severity, sum_loss_costs

collision = frequency_severity(frequency_model, severity_model)
total = sum_loss_costs({"collision": collision, "other_peril": other_loss_cost_model})
quote_costs = total.predict(quotes)
component_costs = total.predict_components(quotes)
total.save("peril_plan")
reloaded = ComposedModel.load("peril_plan")
```

`frequency_severity` multiplies a recorded Poisson rate by a Gamma severity mean.
A count-offset frequency is evaluated as a rate, so quotes do not need exposure and
exposure is not applied twice. The caller must choose compatible claim definitions:
for example, positive-payment severity times all-claim frequency needs explicit
handling of zero-payment claims. Currency, trend level and exposure unit compatibility
are caller assertions, recorded with the result's `unit` (default loss_per_exposure).

`sum_loss_costs` sums the response means of named components. Components may themselves
be compositions. Names, component models, category encodings and units are retained.
It rejects direct frequency rate/count sums masquerading as loss costs, and nested
compositions with different declared units.

A composed mean has `family=None`. It never inherits a likelihood from its first
component. `validate(data, target=..., metric=..., weight=...)` requires an explicit
metric and returns the common comparison exhibits. Convergence is false if any
component failed, true only if every component has recorded convergence, and otherwise
unknown. Loaded workbook components do not invent fitting diagnostics.

`save` writes a versioned composition manifest and independently editable component
workbooks. Nested manifests preserve lineage. `load` rejects unsupported future
versions. This is a scoring composition artifact, not yet the complete analytical
bundle described in the improvement plan. Revalidate after editing component files.

Existing `FittedModel +` behavior is unchanged: it adds linear predictors, multiplying
means under log links. Use the named operations to make multiplication versus
response-scale addition explicit. Merging category encodings now preserves wildcard
sentinels so legacy combined named models retain valid fallback routes.
