"""Peak resident memory of a block of work, shared by the benchmark scripts.

Avenue's claim is that it never builds a design matrix, and a claim about memory has
to be measured rather than argued. All three benchmarks report the same figure, from
the same sampler, so their numbers can be read side by side.
"""

from __future__ import annotations

import gc
import threading


class PeakMemory:
    """Peak resident memory an engine adds, over the baseline when it started.

    Sampled from a background thread rather than read before and after, because the
    interesting number is the high-water mark - a design matrix that is built, used and
    freed still has to fit in RAM.

    Two caveats worth knowing when reading the result. CPython does not always return
    freed memory to the OS, so an engine that runs after a greedy one can show a
    smaller figure than it would alone; and the shared dataset is allocated before any
    of this, so it is excluded from every engine's total.
    """

    INTERVAL_SECONDS = 0.002

    def __init__(self) -> None:
        import psutil

        self._process = psutil.Process()
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0.0
        self._peak_bytes = 0

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = self._process.memory_info().rss
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._stop.wait(self.INTERVAL_SECONDS)

    def __enter__(self) -> "PeakMemory":
        gc.collect()
        self._baseline = self._process.memory_info().rss
        self._peak_bytes = self._baseline
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_mb = max(0.0, (self._peak_bytes - self._baseline) / 1e6)


def measured(fn):
    """Run `fn`, returning its result dict with `peak_mb` filled in."""
    with PeakMemory() as mem:
        result = fn()
    result["peak_mb"] = mem.peak_mb
    return result
