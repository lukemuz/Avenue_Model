# Large numeric table scoring

Avenue recognizes complete numeric Cartesian grids and searches each axis by binary
search. A dense coordinate index identifies the original factor row. The index is
rebuilt for each matching call, so workbook edits cannot leave stale factors or geometry
in a persistent cache. Public prediction still prepares inputs and checks unmatched and
nonfinite results.

For fewer than 32 quote rows, multi-column numeric tables use the direct matcher to
avoid index construction overhead. That scan stops at the first match using zero
categorical wildcards: no later row can improve its specificity, and ties preserve
the first row. Matches using wildcards continue searching for a more specific row.
Tests compare both sides of the batch-size cutoff with the independent row matcher,
including empty and chunked frames. The cutoff is a conservative local heuristic,
not a guarantee of optimal dispatch for every table geometry and machine.

Recognition requires unique complete coordinates, non-null thresholds, and
row order in which each numeric coordinate's immediate predecessor occurs earlier. Transitivity
then proves that the coordinate found by binary search precedes every other matching
row. Explicit missing-only `NaN` bounds use a separate coordinate on each axis,
outside the ordered numeric search. Null quote values and NaNs select that coordinate
when it exists; finite quotes cannot select it. No predecessor relation is imposed
between missing and numeric coordinates. A null *table threshold* still means an
unconstrained bound and prevents this index from being used.

This supports valid orders beyond a particular lexicographic layout. Incomplete,
duplicate, incorrectly ordered and mixed categorical/numeric tables retain the
general matcher. This optimization does not approximate a table or change its factors.

The independent row matcher checks a three-dimensional grid against all 2,197
combinations of boundary, tail, signed-zero, infinite, null and NaN probes, plus invalid
grid variants. Two additional three-dimensional grids check explicit missing routes,
including a missing-only axis, 2,662 probe combinations, NaN payloads and chunked data.
Conversion tests exercise tiny and indexed-size quote batches through both
consolidation modes and workbook reloads.
