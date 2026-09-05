# Per-trial converted artifact measurements

Both stock/fork booster tutorials pass with fold-level row counts and interaction
complexity recorded at the selected boosting prefix. The complete Python suite passes
134 tests. Focused tests force selection of round one from longer real boosters,
compare measurements to converted artifacts, cover constant ensembles, and show that
equal four-table counts with 40 versus 19,181 rows remain distinguishable in summaries.
The 19,181-row summary fixture checks presentation; it is not a new performance benchmark.

The real motor stock study also passes all fitting, independent prediction, conversion,
reload and delivery gates. Its four trials and three folds required 0.885 seconds of
dump/conversion/table extraction within 4.224 seconds of tuning. This extra measurement
cost is explicit and can grow substantially for larger artifacts; it is not scoring cost.
Compared to the preceding wheel acceptance, all trial parameters, selected iterations
and table counts agree, with loss differences below 1e-14.

The selected trial's folds have 11, 11 and 10 tables, with 1,123, 951 and 1,155 rows;
their largest tables have 352, 352 and 378 rows. Maximum interaction order is two.
These are measured CV artifacts, not a guarantee about the final full-data refit.
Statistical rank and support remain unmeasured and are labeled accordingly.

The retained records include complete trial measurements and the real run/result.
`provenance.json` records source and full real-artifact hashes. Full study outputs
remain under `/tmp/avenue-real-trial-complexity`; the tutorial outputs are under
`/tmp/avenue-complexity-{stock,fork}`. Historical wheel acceptance is unchanged; this
Python-only increment was checked from editable source.
