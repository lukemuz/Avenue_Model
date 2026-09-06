# Acceptance scope

Acceptance targets dependable fitting, inference, conversion and editable scoring.
The earlier feature-expansion checklist is superseded by the
[fundamentals scope review](COMPLEXITY_REVIEW.md).

Run the Python regression suite and Rust tests for numerical changes. The
[wheel acceptance runner](WHEEL_ACCEPTANCE.md) verifies the imported artifact and
executes the retained tests and pricing examples. The API builder checks that public
exports have generated documentation anchors.

Real-data evidence is retained in the [motor study](REAL_MOTOR_ACCEPTANCE.md),
[spline study](REAL_SPLINE_ACCEPTANCE.md), [fork study](REAL_FORK_ACCEPTANCE.md),
and [large-table scoring study](LARGE_TABLE_SCORING.md). Historical outputs describe
the exact earlier builds recorded there; their former workflow wrappers are not
part of the current API. Current study scripts use explicit data-frame operations
and scikit-learn splits/metrics while preserving independent prediction checks.

Remaining statistical limitations are documented with the relevant primitive:
[splines](SPLINES.md), [intervals](COEFFICIENT_INTERVALS.md),
[HC0](ROBUST_INFERENCE.md), [CR0](CLUSTER_INFERENCE.md), and
[joint tests](TERM_TESTS.md). These boundaries do not imply a commitment to add
specialized estimators or general experiment-management infrastructure.
