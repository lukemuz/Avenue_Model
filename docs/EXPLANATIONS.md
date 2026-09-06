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
changes = quotes.select("policy_id").with_columns(
    original.predict(quotes).to_series().alias("before"),
    edited.predict(quotes).to_series().alias("after"),
).with_columns((pl.col("after") - pl.col("before")).alias("change"))
```

Compare means in the same physical units and exposure convention. Aggregate with
Polars using exposure weights for rates, or sum totals directly. Inspect each
model's contributions when term-level explanations are needed.

An edited workbook is a new scoring artifact; it does not inherit standard errors
for its edited coefficients. Obtain fresh validation with `edited.report(data)`.
The auto example verifies a known 5% edit and saves policy movements and explanations.
