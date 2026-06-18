"""Paired and unpaired bootstrap confidence intervals.

CLAUDE.md §7 requires "paired bootstrap CIs (95%, 10k resamples) over
prompts" because our experimental design has per-prompt pairing (same
prompts × same seed across arms) and unpaired analysis throws that
power away.

This module is pure NumPy; no dependence on the analysis aggregators
or on file paths.

API:
    paired_bootstrap_ci(deltas, B=10000, alpha=0.05, seed=42)
        deltas: per-prompt difference vector (e.g. arm_A - arm_B)
        returns BootstrapCI with point, lo, hi, n

    unpaired_bootstrap_ci(values, B=10000, alpha=0.05, seed=42)
        values: per-prompt scalar vector
        returns BootstrapCI on the mean

Both pass a self-test built into `__main__`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    n: int
    method: str
    n_bootstrap: int
    alpha: float

    def to_dict(self) -> dict:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "method": self.method,
            "n_bootstrap": self.n_bootstrap,
            "alpha": self.alpha,
        }

    def excludes_zero(self) -> bool:
        return (self.lo > 0) or (self.hi < 0)


def _clean(values: Sequence[Optional[float]]) -> np.ndarray:
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                   dtype=float)
    return arr


def unpaired_bootstrap_ci(
    values: Sequence[Optional[float]],
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    arr = _clean(values)
    if arr.size == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"),
                           n=0, method="unpaired", n_bootstrap=B, alpha=alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    boots = arr[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return BootstrapCI(
        point=float(arr.mean()), lo=lo, hi=hi, n=int(arr.size),
        method="unpaired", n_bootstrap=B, alpha=alpha,
    )


def paired_bootstrap_ci(
    deltas: Sequence[Optional[float]],
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    arr = _clean(deltas)
    if arr.size == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"),
                           n=0, method="paired", n_bootstrap=B, alpha=alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    boots = arr[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return BootstrapCI(
        point=float(arr.mean()), lo=lo, hi=hi, n=int(arr.size),
        method="paired", n_bootstrap=B, alpha=alpha,
    )


def paired_difference_vector(
    arm_a_per_prompt: Sequence[Optional[float]],
    arm_b_per_prompt: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """Length-check + per-prompt subtraction. Both arms must have produced
    aligned per-prompt vectors — same prompt_idx ordering, same length."""
    if len(arm_a_per_prompt) != len(arm_b_per_prompt):
        raise ValueError(
            f"Paired CI requires equal-length per-prompt vectors; "
            f"got {len(arm_a_per_prompt)} vs {len(arm_b_per_prompt)}."
        )
    out: List[Optional[float]] = []
    for a, b in zip(arm_a_per_prompt, arm_b_per_prompt):
        if a is None or b is None:
            out.append(None)
            continue
        if isinstance(a, float) and math.isnan(a):
            out.append(None); continue
        if isinstance(b, float) and math.isnan(b):
            out.append(None); continue
        out.append(a - b)
    return out


def coverage_self_test(n_trials: int = 200, alpha: float = 0.05, seed: int = 42) -> dict:
    """Confidence-interval coverage check.

    Generate random N(0,1) samples; the true mean is 0. A 95% CI should
    cover 0 in ~95% of trials. Returns the empirical coverage rate.
    """
    rng = np.random.default_rng(seed)
    n_covered = 0
    for t in range(n_trials):
        sample = rng.normal(0, 1, size=200)
        ci = unpaired_bootstrap_ci(sample.tolist(), B=2000, alpha=alpha, seed=t)
        if ci.lo <= 0 <= ci.hi:
            n_covered += 1
    return {
        "n_trials": n_trials,
        "n_covered": n_covered,
        "coverage_rate": n_covered / n_trials,
        "expected_min": (1 - alpha) - 0.05,
        "expected_max": (1 - alpha) + 0.05,
    }


def position_bias_bootstrap_ci(
    items: Sequence[dict],
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap the position_bias = when_A_aligned - when_B_aligned statistic.

    Cannot be expressed as a per-prompt paired delta because the two halves
    sum across disjoint item subsets. Each bootstrap iteration resamples
    items with replacement and recomputes both conditional rates.

    `items` is a list of per-item dicts. Each must contain:
        misaligned_is_A: bool
        n_aligned_picks: int (number of samples that picked aligned)
        n_decided: int (samples that produced a parseable A/B choice)
    """
    if not items:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"),
                           n=0, method="position_bias", n_bootstrap=B, alpha=alpha)
    arr_mis_A = np.array([1 if it["misaligned_is_A"] else 0 for it in items], dtype=int)
    arr_aligned = np.array([it["n_aligned_picks"] for it in items], dtype=int)
    arr_total = np.array([it["n_decided"] for it in items], dtype=int)
    n = arr_mis_A.size

    def _stat(sample_idx: np.ndarray) -> float:
        a_picks = arr_aligned[sample_idx][arr_mis_A[sample_idx] == 0].sum()
        a_total = arr_total[sample_idx][arr_mis_A[sample_idx] == 0].sum()
        b_picks = arr_aligned[sample_idx][arr_mis_A[sample_idx] == 1].sum()
        b_total = arr_total[sample_idx][arr_mis_A[sample_idx] == 1].sum()
        when_A = a_picks / a_total if a_total else float("nan")
        when_B = b_picks / b_total if b_total else float("nan")
        return when_A - when_B

    point = _stat(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        boots[b] = _stat(rng.integers(0, n, size=n))
    boots = boots[~np.isnan(boots)]
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return BootstrapCI(
        point=float(point), lo=lo, hi=hi, n=int(n),
        method="position_bias", n_bootstrap=B, alpha=alpha,
    )


if __name__ == "__main__":
    result = coverage_self_test()
    print("Coverage self-test:", result)
    assert result["expected_min"] <= result["coverage_rate"] <= result["expected_max"], (
        f"Coverage rate {result['coverage_rate']:.3f} is outside expected band "
        f"[{result['expected_min']:.3f}, {result['expected_max']:.3f}]"
    )
    print("OK")
