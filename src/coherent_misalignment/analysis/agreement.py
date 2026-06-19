"""Inter-rater agreement metrics for ordinal judge scores.

Lifted and consolidated from `scripts/judge_agreement.py` so multiple
scripts (Phase 1 Claude vs GPT, Phase 7 Gemini vs GPT) can reuse the
same implementation. Adds Krippendorff's α (ordinal) which the
hardening plan calls for as a robustness check against Cohen's κ.

Public API:
    cohens_kappa(y1, y2, weighted=False)
        Unweighted: standard κ.
        weighted=True: quadratic weights for ordinal labels.

    krippendorffs_alpha(y1, y2, level="ordinal")
        Krippendorff's α. "nominal" and "ordinal" supported.

    pearson_r(x, y)
        Pearson correlation.

    confusion_matrix(y1, y2, labels)
        Square matrix of co-occurrences.

    per_bucket_agreement(y1, y2, n_buckets=5)
        Per-label exact-match and within-1 rates. Useful for stratifying
        the headline κ by harm-score bucket.

    agreement_summary(y1, y2, threshold=4)
        Bundle every metric above into one dict.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def cohens_kappa(y1: Sequence[int], y2: Sequence[int], weighted: bool = False) -> float:
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have the same length")
    if not y1:
        return float("nan")
    labels = sorted(set(y1) | set(y2))
    k = len(labels)
    if k < 2:
        return 1.0 if list(y1) == list(y2) else 0.0
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        cm[idx[a]][idx[b]] += 1
    n = sum(sum(r) for r in cm)
    row = [sum(r) for r in cm]
    col = [sum(cm[i][j] for i in range(k)) for j in range(k)]

    if weighted:
        span_sq = (max(labels) - min(labels)) ** 2
        if span_sq == 0:
            return 1.0
        w = [
            [((labels[i] - labels[j]) ** 2) / span_sq for j in range(k)]
            for i in range(k)
        ]
        p_o_w = sum(w[i][j] * cm[i][j] for i in range(k) for j in range(k)) / n
        p_e_w = sum(w[i][j] * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
        return 1.0 - (p_o_w / p_e_w) if p_e_w > 0 else float("nan")

    p_o = sum(cm[i][i] for i in range(k)) / n
    p_e = sum(row[i] * col[i] for i in range(k)) / (n * n)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def krippendorffs_alpha(
    y1: Sequence[int], y2: Sequence[int], level: str = "ordinal"
) -> float:
    """Krippendorff's α for two raters, no missing values.

    α = 1 − D_o / D_e, where D_o is observed disagreement and D_e is
    expected disagreement under random pairing.

    Distance functions per Krippendorff (2011):
      - nominal: 1 if a ≠ b else 0
      - ordinal: ((b_rank − a_rank) − (counts_between/2))^2 in ranks
                 — implemented here as squared rank distance scaled by
                 the standard ordinal-α formula.
      - interval: (a − b)^2

    For ordinal-level agreement on a 1-5 harm scale, this gives a
    different penalty profile than quadratic-weighted κ:
    Krippendorff penalises distance by *cumulative-rank distance*, not
    raw label-distance squared.
    """
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have the same length")
    n = len(y1)
    if n == 0:
        return float("nan")

    labels = sorted(set(y1) | set(y2))
    if len(labels) < 2:
        return 1.0

    pairs = list(zip(y1, y2))
    label_freq: Dict[int, int] = {l: 0 for l in labels}
    for a, b in pairs:
        label_freq[a] += 1
        label_freq[b] += 1
    n_pairings = 2 * n  # each rater contributes one value per unit

    if level == "nominal":
        def dist(a: int, b: int) -> float:
            return 0.0 if a == b else 1.0
    elif level == "interval":
        def dist(a: int, b: int) -> float:
            return float((a - b) ** 2)
    elif level == "ordinal":
        cum_counts: Dict[int, int] = {}
        running = 0
        for lab in labels:
            cum_counts[lab] = running + label_freq[lab] / 2
            running += label_freq[lab]

        def dist(a: int, b: int) -> float:
            if a == b:
                return 0.0
            if a > b:
                a, b = b, a
            between = sum(label_freq[l] for l in labels if a < l < b)
            d = (label_freq[a] + label_freq[b]) / 2 + between
            return float(d * d)
    else:
        raise ValueError(f"unknown level: {level!r}")

    d_o_numer = sum(dist(a, b) for a, b in pairs)
    d_o = d_o_numer / n

    d_e_numer = 0.0
    for la in labels:
        for lb in labels:
            d_e_numer += label_freq[la] * label_freq[lb] * dist(la, lb)
    denom_factor = n_pairings * (n_pairings - 1)
    if denom_factor <= 0:
        return float("nan")
    d_e = d_e_numer / denom_factor

    if d_e <= 0:
        return 1.0 if d_o == 0 else float("nan")
    return 1.0 - (d_o / d_e)


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def confusion_matrix(
    y1: Sequence[int], y2: Sequence[int], labels: Sequence[int]
) -> List[List[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        cm[idx[a]][idx[b]] += 1
    return cm


def per_bucket_agreement(
    y1: Sequence[int], y2: Sequence[int], n_buckets: int = 5
) -> Dict[int, Dict[str, float]]:
    """For each judge-1 bucket, report: n, exact-match rate, within-1 rate,
    judge-2 mean. Helps localize disagreement (e.g. "score=3 is where
    judges disagree most")."""
    out: Dict[int, Dict[str, float]] = {}
    for b in range(1, n_buckets + 1):
        idxs = [i for i, v in enumerate(y1) if v == b]
        if not idxs:
            out[b] = {"n": 0, "exact_rate": float("nan"), "within_1_rate": float("nan"), "y2_mean": float("nan")}
            continue
        exact = sum(1 for i in idxs if y1[i] == y2[i]) / len(idxs)
        within1 = sum(1 for i in idxs if abs(y1[i] - y2[i]) <= 1) / len(idxs)
        y2_mean = sum(y2[i] for i in idxs) / len(idxs)
        out[b] = {"n": len(idxs), "exact_rate": exact, "within_1_rate": within1, "y2_mean": y2_mean}
    return out


def agreement_summary(
    y1: Sequence[int], y2: Sequence[int], threshold: int = 4, labels: Sequence[int] = (1, 2, 3, 4, 5)
) -> Dict[str, object]:
    """One-shot bundle of every agreement metric for two raters' 1-5 scores."""
    if not y1 or not y2 or len(y1) != len(y2):
        return {"n": 0}
    n = len(y1)
    exact = sum(a == b for a, b in zip(y1, y2)) / n
    within1 = sum(abs(a - b) <= 1 for a, b in zip(y1, y2)) / n
    bin1 = [1 if s >= threshold else 0 for s in y1]
    bin2 = [1 if s >= threshold else 0 for s in y2]
    return {
        "n": n,
        "cohens_kappa_unweighted": cohens_kappa(y1, y2, weighted=False),
        "cohens_kappa_quadratic_weighted": cohens_kappa(y1, y2, weighted=True),
        "krippendorffs_alpha_ordinal": krippendorffs_alpha(y1, y2, level="ordinal"),
        "krippendorffs_alpha_nominal": krippendorffs_alpha(y1, y2, level="nominal"),
        "pearson_r": pearson_r([float(s) for s in y1], [float(s) for s in y2]),
        "exact_agreement_rate": exact,
        "within_1_agreement_rate": within1,
        "binary_kappa": cohens_kappa(bin1, bin2, weighted=False),
        "binary_agreement_rate": sum(a == b for a, b in zip(bin1, bin2)) / n,
        "y1_mean": sum(y1) / n,
        "y2_mean": sum(y2) / n,
        "confusion_matrix": {
            "labels": list(labels),
            "axis0": "rater_1 (row)",
            "axis1": "rater_2 (col)",
            "counts": confusion_matrix(y1, y2, list(labels)),
        },
        "per_bucket": per_bucket_agreement(y1, y2),
        "threshold": threshold,
    }
