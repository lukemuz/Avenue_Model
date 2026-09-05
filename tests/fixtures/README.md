# Conversion regression fixtures

`categorical_tree.json` is the first failing tree isolated in
`evaluation/fork_results/conversion_diagnostics.json` by the September 2026 fork
conversion evaluation. It contains model structure only, with no policy records.
The regression test enumerates a small synthetic Cartesian product of its drivers
and evaluates the original tree decisions independently of Avenue.

`test_conversion_contract.py` additionally constructs minimal two-category and
adjacent-floating-point threshold fixtures, deterministic randomized trees, and a
small booster trained by the installed LightGBM build. Each checks both consolidation
modes and a CSV workbook reload at `atol=1e-12, rtol=1e-12`. The handcrafted fixtures
use distinct leaf values to make incorrect branch membership observable directly.
The suite also covers null/NaN routes, both default directions, repeated numerical
splits and JSON reloads. zero_as_missing remains explicitly unsupported.
