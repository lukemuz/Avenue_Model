# Explicit pandas input adapter

Install `avenue_model[pandas]`, then convert once at the boundary:

```python
from avenue_model import Plan, from_pandas

training = from_pandas(pandas_training, index_column="source_row")
model = Plan.frequency("exposure").categorical("region").fit(training, "frequency")
quotes = from_pandas(pandas_quotes)
predictions = model.predict(quotes)
```

A pandas category's actual labels are retained, not its positional `.cat.codes`.
String, integer and boolean category labels are supported. Reordering the category
list does not change label identity. Unused levels do not become training observations
or enter Avenue's fitted encoding. Nulls, `pd.NA` and floating NaNs become Polars nulls;
nullable integer/boolean columns preserve their values. The input frame is not mutated.

The adapter excludes the pandas index by default. Supply `index_column` to preserve a
single index in a new named column. Reset a MultiIndex into explicit columns yourself.
Output predictions are positional; retain this source identifier when joining back to
other datasets. Column names must be unique strings.

Object columns must contain strings/nulls; cast numerical/date object columns to a
concrete dtype before conversion. Mixed-type and float-valued category labels require
an explicit encoding rather than automatic stringification. That avoids collisions
such as integer `1` and string `"1"`. General Avenue term requirements still apply:
for example a float driver needs a numeric term, not an integer-category interpretation.
Polars remains native; existing methods do not silently convert arbitrary objects.
