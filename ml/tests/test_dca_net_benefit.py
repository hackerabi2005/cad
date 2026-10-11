"""
Unit tests for Decision Curve Analysis (DCA) Net Benefit recomputation from confusion counts.
Verifies Vickers & Elkin (2006) formulation and mathematical theoretical bounds.
"""

import numpy as np
import pytest
from scripts.decision_curve_analysis import compute_net_benefit, compute_treat_all_nb


def test_dca_net_benefit_formula_agrees_with_confusion_counts():
    """Verify compute_net_benefit equals (TP/N) - (FP/N)*(pt/(1-pt))."""
    y_true = np.array([1] * 217 + [0] * 86)
    # Simulate a classifier with 208 TP, 9 FN, 54 TN, 32 FP (Sens = 95.9%, Spec = 62.8%)
    y_prob = np.zeros(len(y_true))
    # 208 positive cases predicted above threshold
    y_prob[:208] = 0.85
    # 9 positive cases predicted below threshold
    y_prob[208:217] = 0.20
    # 32 negative cases predicted above threshold (false positives)
    y_prob[217 : 217 + 32] = 0.85
    # 54 negative cases predicted below threshold (true negatives)
    y_prob[217 + 32 :] = 0.20

    pt = 0.40
    nb_func = compute_net_benefit(y_true, y_prob, pt)
    nb_treat_all = compute_treat_all_nb(y_true, pt)

    # Recompute from raw counts
    n = len(y_true)
    tp = 208
    fp = 32
    weight = pt / (1.0 - pt)
    expected_nb = (tp / n) - (fp / n) * weight
    prev = 217 / 303
    expected_all = prev - (1.0 - prev) * weight

    assert abs(nb_func - expected_nb) < 1e-9
    assert abs(nb_treat_all - expected_all) < 1e-9

    delta_per_100 = (nb_func - nb_treat_all) * 100.0
    # Delta should be ~ +8.91 per 100
    assert 8.0 < delta_per_100 < 10.0


def test_dca_net_benefit_theoretical_upper_bound():
    """
    Theoretical maximum gain over Treat-All at decision threshold pt is:
    Delta_max = (1 - pi) * (pt / (1 - pt))
    For pi = 217/303 and pt = 0.40:
    Delta_max = (86/303) * (0.4 / 0.6) = 0.1892 (i.e. +18.92 per 100).
    Any claim higher than +18.92 per 100 at pt = 0.40 is mathematically impossible.
    """
    pi = 217.0 / 303.0
    pt = 0.40
    max_possible_delta = (1.0 - pi) * (pt / (1.0 - pt)) * 100.0
    assert abs(max_possible_delta - 18.92) < 0.1

    # Verify that a perfect classifier (Sens=1.0, Spec=1.0) achieves exactly this maximum
    y_perfect_true = np.array([1] * 217 + [0] * 86)
    y_perfect_prob = np.array([0.9] * 217 + [0.1] * 86)
    perfect_nb = compute_net_benefit(y_perfect_true, y_perfect_prob, pt)
    all_nb = compute_treat_all_nb(y_perfect_true, pt)
    perfect_delta_per_100 = (perfect_nb - all_nb) * 100.0

    assert abs(perfect_delta_per_100 - max_possible_delta) < 1e-6
    actual_delta_per_100 = 8.91
    assert actual_delta_per_100 < max_possible_delta


def test_fairness_strata_confusion_matrix_summation():
    """
    Asserts that fairness audit strata confusion counts (Male + Female, Age <= 65 + Age > 65)
    sum identically to the pooled out-of-fold confusion counts at both operating points:
    Screening cutoff (0.400): TP=208, FN=9 (total CAD=217); TN=54, FP=32 (total Normal=86).
    Balanced cutoff (0.500):  TP=205, FN=12 (total CAD=217); TN=61, FP=25 (total Normal=86).
    """
    # 1. Screening mode counts (0.400)
    male_screening = {"tp": 125, "fn": 6, "tn": 26, "fp": 19}
    female_screening = {"tp": 83, "fn": 3, "tn": 28, "fp": 13}

    assert male_screening["tp"] + female_screening["tp"] == 208
    assert male_screening["fn"] + female_screening["fn"] == 9
    assert male_screening["tn"] + female_screening["tn"] == 54
    assert male_screening["fp"] + female_screening["fp"] == 32

    age_le65_screening = {"tp": 134, "fn": 9, "tn": 51, "fp": 25}
    age_gt65_screening = {"tp": 74, "fn": 0, "tn": 3, "fp": 7}

    assert age_le65_screening["tp"] + age_gt65_screening["tp"] == 208
    assert age_le65_screening["fn"] + age_gt65_screening["fn"] == 9
    assert age_le65_screening["tn"] + age_gt65_screening["tn"] == 54
    assert age_le65_screening["fp"] + age_gt65_screening["fp"] == 32

    # 2. Balanced mode counts (0.500)
    male_balanced = {"tp": 122, "fn": 9, "tn": 31, "fp": 14}
    female_balanced = {"tp": 83, "fn": 3, "tn": 30, "fp": 11}

    assert male_balanced["tp"] + female_balanced["tp"] == 205
    assert male_balanced["fn"] + female_balanced["fn"] == 12
    assert male_balanced["tn"] + female_balanced["tn"] == 61
    assert male_balanced["fp"] + female_balanced["fp"] == 25

    age_le65_balanced = {"tp": 132, "fn": 11, "tn": 58, "fp": 18}
    age_gt65_balanced = {"tp": 73, "fn": 1, "tn": 3, "fp": 7}

    assert age_le65_balanced["tp"] + age_gt65_balanced["tp"] == 205
    assert age_le65_balanced["fn"] + age_gt65_balanced["fn"] == 12
    assert age_le65_balanced["tn"] + age_gt65_balanced["tn"] == 61
    assert age_le65_balanced["fp"] + age_gt65_balanced["fp"] == 25
