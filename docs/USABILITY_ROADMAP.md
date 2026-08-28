# Avenue Model usability roadmap

## Product direction

Avenue's computational core is strong, but its Python API currently exposes the
engine's internal representation too directly. Avenue should keep rating tables as its
foundation while adding a higher-level modelling interface designed around the workflow
of an insurance data scientist.

The product goal is:

> Easy to fit like a modern GLM package, unusually good at producing deployable rating
> tables.

## Current sources of friction

A basic fit currently requires users to:

- manually construct an intercept table;
- manually create every categorical level and numeric band;
- encode categoricals as `Int32`;
- encode numeric bands as `Float64` with a final `inf`;
- know that `Rating_Factor` is on the linear-predictor scale;
- create `log_exposure` themselves;
- specify the objective on both `RatingModel` and `GLMOptions`;
- work with table indices rather than names; and
- manually align standard errors and diagnostics with table rows.

The most important API issues are:

1. The model controls the link while `GLMOptions` controls the likelihood. These can
   disagree, and omitted options currently default to Gaussian independently of the
   model.
2. Insurance users generally want a base rate and multiplicative relativities, while
   Avenue exposes additive `Rating_Factor` values on the link scale.
3. String and categorical columns are not first-class inputs.
4. Results are fragmented across a fitted model, unnamed DataFrames, and index-based
   diagnostic arrays.
5. Prediction does not distinguish rate, expected claim count, severity, and pure
   premium.
6. Avenue fits existing tables well but does not conveniently create them from ordinary
   modelling data.
7. One-way analysis does not yet provide the complete actual-versus-expected summaries
   used in pricing work.

## Intended public API

### `GLM`: user-facing estimator

```python
from avenue_model import GLM

model = GLM(
    family="poisson",
    terms={
        "driver_age": {"bins": [18, 21, 25, 30, 40, 55, 70]},
        "vehicle_age": {"bins": "quantile", "n_bins": 10},
        "region": "categorical",
        ("vehicle_group", "region"): "interaction",
    },
)

result = model.fit(train, target="claim_count", exposure="exposure")
```

This layer should create the intercept, construct and validate tables, accept ordinary
dataframe types, apply exposure correctly, choose sensible references, and return a
cohesive result object. A formula interface can be added as a convenience over the
structured term specification.

### `GLMResult`: fitted statistical result

```python
result.converged
result.summary()
result.diagnostics
result.rating_tables()
result.predict(test)
result.predict_rate(test)
result.predict_expected(test, exposure="exposure")
result.actual_vs_expected(test, by="region", exposure="exposure")
```

The existing functional fitting API should remain available as the low-level API.

### `RatingModel`: deployment representation

`RatingModel` remains the table-native scoring model, the output of conventional GLM
fitting, the target of LightGBM conversion, and the representation used for composition,
manual adjustment, export, and deployment.

```text
Conventional GLM --\
                   +--> RatingModel --> inspect / adjust / export / deploy
LightGBM ----------/
```

## Rating-table experience

Rating tables should become Avenue's standout user-facing feature:

```python
tables = result.rating_tables(
    scale="relativity",
    include=[
        "estimate", "std_error", "lower", "upper",
        "exposure", "actual", "expected", "observations",
    ],
)
```

Tables should:

- include coefficient and relativity where applicable;
- have stable names;
- identify base, aliased, locked, wildcard, and no-data rows;
- contain diagnostics as columns rather than parallel nested lists;
- state interval semantics clearly;
- be available as a mapping keyed by name or in long form; and
- treat rounding as presentation or export rather than model mutation.

For log-link families, expose the base frequency, severity, or pure premium naturally.

## Insurance conveniences

Provide explicit presets:

```python
GLM.frequency(...)
GLM.severity(...)
GLM.pure_premium(...)
GLM.claim_probability(...)
```

- Frequency: Poisson with log exposure offset.
- Severity: Gamma, commonly weighted by claim count.
- Pure premium: Tweedie with exposure handling.
- Claim probability: Binomial with logit link.

Longer term, support an intuitive two-part frequency-severity workflow and composition
into technical price.

## Safer data handling

Normalize ordinary dataframe types at the API boundary:

- accept pandas and Polars;
- accept strings, enums, categoricals, common integer widths, and floats;
- retain category mappings after internal encoding;
- define explicit missing-value policies;
- validate band ordering, overlap, gaps, and terminal coverage;
- report unmatched values and counts; and
- provide `model.validate(df)` and `model.data_requirements()`.

Strict validation should remain, but errors should explain how to repair input.

## Named terms and table builders

Expose structured builders such as `Categorical`, `Banded`, `Interaction`, and
`Variate`, including data-driven quantile, equal-width, and top-category constructors.
Reference selection should accept a level value rather than depend on row ordering.

Table names must flow through fitting, diagnostics, result presentation, and
serialization.

## Diagnostics and model review

`GLMResult.summary()` should translate numerical diagnostics into actionable messages,
including convergence, empty levels, aliasing, penalized-inference limitations, and poor
conditioning.

Planned statistical additions include:

- likelihood-ratio and whole-term tests;
- robust or sandwich covariance;
- direct confidence intervals;
- deviance residuals and observation diagnostics;
- cross-validation for penalties, Tweedie power, and variate degree;
- credibility or partial-pooling shrinkage; and
- monotonic constraints.

## Prediction API

Prediction semantics should be explicit:

```python
result.predict(df)                         # fitted mean
result.predict_linear(df)                  # linear predictor
result.predict_rate(df)                    # frequency rate
result.predict_expected(df, exposure=...)  # rate * exposure
result.predict_components(df)              # contribution of each table
```

Component prediction should make pricing decisions inspectable as a base value and a
sequence of additive effects or multiplicative relativities.

## Persistence and governance

Provide a stable, versioned artifact carrying family, link, table matching semantics,
category mappings, bins, references, coefficients, locks, diagnostics, package version,
training schema, and optional business metadata. Support JSON and insurance-friendly
Excel exports.

## LightGBM integration

LightGBM conversion should feed the same table-review tools without dominating the basic
GLM API. It should accept a Booster directly and produce a conversion report containing
prediction parity, generated table sizes, represented interactions, missing-category
behavior, and warnings about table explosion.

## Implementation phases

### Phase 1: fix API hazards

1. Remove duplicate family specification and reject family/link inconsistencies.
2. Add named tables and name-based lookup.
3. Introduce `GLMResult`.
4. Return joined, presentation-ready rating tables.
5. Add `predict_linear`, `predict_rate`, `predict_expected`, and
   `predict_components`.
6. Improve exception classes and validation messages.
7. Accept ordinary integer widths and categorical/string data.

### Phase 2: make fitting tables easy

1. Add categorical, banded, interaction, and variate specifications.
2. Create the intercept automatically.
3. Select reference levels by value.
4. Add manual, quantile, and equal-width band builders.
5. Support pandas alongside Polars.
6. Add frequency, severity, and pure-premium presets.
7. Rewrite the quick start around the estimator API.

### Phase 3: make rating-table review exceptional

1. Rich rating-table output.
2. Actual-versus-expected summaries.
3. Confidence intervals and row-status flags.
4. Term summaries and joint tests.
5. Component predictions.
6. Excel and JSON exports.
7. Optional plotting integrations that retain data-first outputs.

### Phase 4: production readiness

1. Stable artifact serialization.
2. Schema and scoring validation.
3. Unseen-category policies.
4. Model metadata and versioning.
5. Batch-scoring contracts.
6. Direct LightGBM Booster conversion and parity reports.
7. Scikit-learn-compatible methods where appropriate.

### Phase 5: advanced actuarial modelling

1. Credibility or hierarchical shrinkage.
2. Monotonicity and shape constraints.
3. Robust covariance.
4. Cross-validation and regularization paths.
5. Frequency-severity composition.
6. Model comparison and term selection.

## Documentation plan

The primary documentation should follow the user workflow:

1. Five-minute insurance frequency example.
2. Reading and exporting rating tables.
3. Severity and pure-premium examples.
4. Creating bands, categoricals, and interactions.
5. Diagnostics and model comparison.
6. Prediction and deployment.
7. LightGBM conversion.
8. Advanced controls.
9. Solver architecture and benchmarks.

A complete freMTPL2 tutorial should cover preparation, fitting, validation, rating-factor
inspection, actual-versus-expected analysis, interaction refinement, table export, and
out-of-sample evaluation.

## Success criterion

An insurance data scientist should be able to go from an ordinary dataframe to reviewed
rating tables with code this short:

```python
result = GLM.frequency(
    terms=[
        Banded("driver_age", breaks=[18, 21, 25, 30, 40, 55, 70]),
        Categorical("region"),
        Categorical("vehicle_group"),
    ]
).fit(train, target="claim_count", exposure="exposure")

result.summary()
result.rating_tables(scale="relativity")
result.actual_vs_expected(validation, by="region", exposure="exposure")
```

Avenue's performance, exact table representation, and LightGBM conversion should feel
like benefits rather than prerequisites.
