# Motor numerical comparison

These compact records were produced during the fundamentals reduction preceding
commit c3fe4b2 on Linux / CPython 3.12. `run.json` records script/input hashes,
dependency versions and limitations. `models.json` records each original fit and
independent glum comparison. All six family/specification checks have zero failed
holdout predictions at `atol=1e-6, rtol=2e-6`.

The banded and five-quantile-knot spline runs use the same grouped holdout and paid
claim definition. Their `comparison.csv` files evaluate predictions with a common
Tweedie power of 1.5. For these specifications, the banded models have lower holdout
loss. This is one split and one fixed knot count, not a general verdict on splines.

The banded run also exercised the stock-LightGBM challenger. Its comparison row
records a short search, not exhaustive tuning. Detailed generated workbooks/reports
are reproduced by the [current study commands](../../README.md#real-motor-comparison);
those bulky artifacts are not required by the regression suite.

These are historical results for the recorded build. Script cleanup may change
script hashes without changing the numerical checks. New measurements belong in a
new output directory; do not overwrite these records and keep their old provenance.
