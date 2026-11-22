"""Choosing the 4 s training window from a recording (ShotShield v4).

Recordings range from 1 s (C3GD) to minutes (field recordings). A clean copy uses the k-th loudest
non-overlapping window (k = 0 for one-event files; k > 0 to take several bangs from one firecracker
recording). An augmented copy moves the event to a random position inside the window, because the live
demo's sliding window sees a sound at any position, not only centred.
"""
from __future__ import annotations

import numpy as np

SILENCE = 2e-5   # below one 16-bit step: only digital silence is trimmed


def trim_digital_silence(y: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(np.abs(y) > SILENCE)
    return y[idx[0]: idx[-1] + 1] if len(idx) else y[:0]


def window_energies(y: np.ndarray, n: int, step: int):
    c = np.concatenate([[0.0], np.cumsum(np.square(y, dtype=np.float64))])
    starts = np.arange(0, len(y) - n + 1, step)
    return starts, c[starts + n] - c[starts]


def kth_loudest_start(y: np.ndarray, n: int, k: int = 0, step: int = 512, min_ratio: float = 0.02):
    """Start of the k-th loudest window that does not overlap louder ones, or None if there is none.

    Windows much quieter than the loudest (energy < min_ratio of it) do not count as another event.
    """
    if len(y) <= n:
        return 0 if k == 0 else None
    starts, e = window_energies(y, n, step)
    e = e.copy()
    top = e.max()
    chosen = None
    for _ in range(k + 1):
        i = int(np.argmax(e))
        if e[i] <= 0 or (chosen is not None and e[i] < min_ratio * top):
            return None
        chosen = int(starts[i])
        e[np.abs(starts - chosen) < n] = -1
    return chosen


def random_event_crop(y: np.ndarray, n: int, start: int, rng, margin: float = 0.1) -> np.ndarray:
    """Re-crop around the loudest point of window [start, start+n) so that it lands at a random position."""
    peak = start + int(np.argmax(np.abs(y[start:start + n])))
    pos = int(rng.uniform(margin, 1 - margin) * n)            # where the peak will be inside the window
    s = int(np.clip(peak - pos, 0, len(y) - n))
    return y[s:s + n]
