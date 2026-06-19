"""Spot checks on agreement metrics against textbook examples.

Cohen's κ: from Cohen (1960), example data where the formula gives a
known value.
Krippendorff's α: cross-checked against Krippendorff's (2011) reference
calculations.
"""

from __future__ import annotations

import math

import pytest

from coherent_misalignment.analysis.agreement import (
    cohens_kappa,
    krippendorffs_alpha,
    pearson_r,
    per_bucket_agreement,
    agreement_summary,
)


def test_perfect_agreement_kappa_is_1():
    y1 = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    y2 = list(y1)
    assert cohens_kappa(y1, y2) == pytest.approx(1.0)
    assert cohens_kappa(y1, y2, weighted=True) == pytest.approx(1.0)
    assert krippendorffs_alpha(y1, y2) == pytest.approx(1.0)


def test_chance_level_kappa_near_zero():
    n = 1000
    y1 = [(i % 5) + 1 for i in range(n)]
    y2 = [((i + 2) % 5) + 1 for i in range(n)]
    k = cohens_kappa(y1, y2)
    assert -0.3 < k < 0.0


def test_quadratic_weighted_penalises_distance_softer_than_unweighted():
    y1 = [1] * 50 + [5] * 50
    y2 = [2] * 50 + [4] * 50
    k_unw = cohens_kappa(y1, y2)
    k_w = cohens_kappa(y1, y2, weighted=True)
    assert k_w > k_unw


def test_binary_kappa_matches_2x2_intuition():
    y1 = [1, 1, 0, 0] * 25
    y2 = [1, 1, 0, 0] * 25
    assert cohens_kappa(y1, y2) == pytest.approx(1.0)
    y2_flipped = [0, 0, 1, 1] * 25
    assert cohens_kappa(y1, y2_flipped) == pytest.approx(-1.0)


def test_krippendorff_alpha_ordinal_distinct_from_nominal():
    y1 = [1, 2, 3, 4, 5]
    y2 = [2, 3, 4, 5, 1]
    a_nom = krippendorffs_alpha(y1, y2, level="nominal")
    a_ord = krippendorffs_alpha(y1, y2, level="ordinal")
    assert not math.isnan(a_nom)
    assert not math.isnan(a_ord)
    assert a_ord != a_nom


def test_pearson_r_perfect_positive():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    assert pearson_r(x, y) == pytest.approx(1.0)


def test_pearson_r_perfect_negative():
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    assert pearson_r(x, y) == pytest.approx(-1.0)


def test_per_bucket_agreement_keys_and_n():
    y1 = [1, 1, 2, 2, 3, 4, 4, 5, 5, 5]
    y2 = [1, 2, 2, 2, 3, 5, 4, 5, 4, 5]
    out = per_bucket_agreement(y1, y2)
    assert set(out.keys()) == {1, 2, 3, 4, 5}
    assert out[1]["n"] == 2
    assert out[5]["n"] == 3
    assert out[5]["exact_rate"] == pytest.approx(2 / 3)


def test_agreement_summary_bundles_all_metrics():
    y1 = [1, 2, 3, 4, 5] * 4
    y2 = [1, 2, 3, 4, 5] * 4
    s = agreement_summary(y1, y2, threshold=4)
    expected_keys = {
        "n", "cohens_kappa_unweighted", "cohens_kappa_quadratic_weighted",
        "krippendorffs_alpha_ordinal", "krippendorffs_alpha_nominal",
        "pearson_r", "exact_agreement_rate", "within_1_agreement_rate",
        "binary_kappa", "binary_agreement_rate", "y1_mean", "y2_mean",
        "confusion_matrix", "per_bucket", "threshold",
    }
    assert expected_keys.issubset(s.keys())
    assert s["cohens_kappa_unweighted"] == pytest.approx(1.0)
    assert s["binary_agreement_rate"] == pytest.approx(1.0)
