# Avenue user guides

Start with the [README example](../README.md#try-it). The public workflow
has three objects: `Plan` defines what to fit, `FittedModel` predicts, refits and reports,
and `Workbook` stores editable tables.

| Guide | Covers |
|---|---|
| [Installation and development](installation.md) | Release wheels, source builds and publishing |
| [Model specification](modeling.md) | Existing plan import/refit/offset workflows, interactions and smooth effects |
| [Scoring and model review](scoring.md) | Exposure conventions, pandas input, band bounds and quote explanations |
| [Statistical inference](inference.md) | Coefficient intervals, HC0, clustered covariance and whole-term tests |
| [LightGBM as rating tables](lightgbm.md) | Sparsity penalties, tuning, exact conversion and GLM refitting |

Use ordinary Polars, NumPy and scikit-learn operations for preparation, validation
splits and model comparisons. The [auto](../examples/auto_pricing_study.py),
[homeowners](../examples/homeowners_perils.py),
[introductory boosting](../examples/interpretable_boosting.py) and
[spline](../examples/smooth_pricing_study.py) examples show complete workflows.

Workbooks preserve scoring. Keep the source Plan, fit options, reports and inference
results with your experiment records, and revalidate edited models.

For implementation and reproduction details, see the
[GLM solver](../src/glm/README.md),
[performance results](performance.md),
[table representation](../src/rating_model/README.md),
[large-table scoring](LARGE_TABLE_SCORING.md), and
[evaluation guide](../studies/README.md). The [API reference guide](API_REFERENCE.md)
explains how to generate searchable signatures and docstrings.
