"""
Progress reporting for long stages.

Extraction walks tens of thousands of frames and preprocessing fits a
baseline per fiber; both take minutes, and a silent terminal is
indistinguishable from a hang.  ``tqdm`` is used when installed and a plain
periodic line otherwise, so the pipeline never depends on it.

Output goes to stderr so that a redirected stdout log stays clean.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Optional


def _tqdm():
    try:
        from tqdm.auto import tqdm
        return tqdm
    except Exception:
        return None


class Stage:
    """A named step with optional progress, usable as a context manager.

    ``with Stage("extract", total=n) as st: ... st.update(1)``
    """

    def __init__(self, label: str, total: Optional[int] = None,
                 enabled: bool = True, stream=sys.stderr,
                 min_interval_s: float = 2.0):
        self.label = label
        self.total = total
        self.enabled = enabled
        self.stream = stream
        self.min_interval = min_interval_s
        self._n = 0
        self._t0 = None
        self._last = 0.0
        self._bar = None

    def __enter__(self) -> "Stage":
        self._t0 = self._last = time.monotonic()
        if not self.enabled:
            return self
        T = _tqdm()
        if T is not None and self.total:
            self._bar = T(total=self.total, desc=self.label, unit="frame",
                          leave=False, file=self.stream,
                          bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                                     "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        else:
            print(f"  {self.label} ...", file=self.stream, flush=True)
        return self

    @property
    def n(self) -> int:
        """Items counted so far."""
        return self._n

    def update_to(self, k: int) -> None:
        """Advance to an absolute count (for callbacks that report totals)."""
        self.update(max(0, int(k) - self._n))

    def update(self, n: int = 1) -> None:
        self._n += n
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.update(n)
            return
        now = time.monotonic()
        if now - self._last < self.min_interval:
            return
        self._last = now
        el = now - self._t0
        if self.total:
            frac = self._n / self.total
            eta = el / frac - el if frac > 0 else float("nan")
            print(f"  {self.label}: {self._n}/{self.total} "
                  f"({frac:.0%}) {el:.0f}s elapsed, ~{eta:.0f}s left",
                  file=self.stream, flush=True)
        else:
            print(f"  {self.label}: {self._n} ({el:.0f}s)",
                  file=self.stream, flush=True)

    def set_total(self, total: int) -> None:
        self.total = total
        if self._bar is not None:
            self._bar.total = total
            self._bar.refresh()

    def __exit__(self, *exc) -> None:
        if self._bar is not None:
            self._bar.close()
        if self.enabled and exc[0] is None:
            el = time.monotonic() - (self._t0 or time.monotonic())
            n = f"{self._n} " if self._n else ""
            print(f"  {self.label}: done ({n}in {el:.1f}s)",
                  file=self.stream, flush=True)


def iter_progress(it: Iterable, label: str, total: Optional[int] = None,
                  enabled: bool = True):
    """Wrap an iterable in a Stage."""
    with Stage(label, total=total, enabled=enabled) as st:
        for x in it:
            yield x
            st.update(1)
