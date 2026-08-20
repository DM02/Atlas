"""Latency metrics: p50/p95 over a set of per-query latency samples (seconds)."""

from __future__ import annotations

import math


def percentile(samples: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 100]. No interpolation between ranks
    — the simplest correct definition, adequate for benchmark reporting here.
    """
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = math.ceil(p / 100 * len(ordered)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def p50(samples: list[float]) -> float:
    return percentile(samples, 50)


def p95(samples: list[float]) -> float:
    return percentile(samples, 95)
