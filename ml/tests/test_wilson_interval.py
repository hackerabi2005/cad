import pytest
from ml.service import compute_wilson_score_interval, compute_similar_cohort_summary


def test_hand_computed_wilson_interval():
    """
    Verifies that compute_wilson_score_interval matches hand-computed analytical values
    for binomial proportions at 95% confidence level.
    """
    # Case 1: k=40, n=50 (80% observed positive rate)
    # Hand calculation:
    # z = 1.959963984540054, z^2 = 3.8414588
    # denom = 1 + 3.8414588/50 = 1.076829
    # center = (0.8 + 3.8414588/100) / 1.076829 = 0.778596
    # margin = z * sqrt(0.8*0.2/50 + 3.8414588/10000) / 1.076829 = 0.108967
    # lower = 0.778596 - 0.108967 = 0.6696
    # upper = 0.778596 + 0.108967 = 0.8876
    low, high = compute_wilson_score_interval(40, 50)
    assert low == 0.6696
    assert high == 0.8876

    # Case 2: Boundary case k=0, n=50 (0% observed)
    low_0, high_0 = compute_wilson_score_interval(0, 50)
    assert low_0 == 0.0
    assert high_0 == 0.0713

    # Case 3: Boundary case k=50, n=50 (100% observed)
    low_50, high_50 = compute_wilson_score_interval(50, 50)
    assert low_50 == 0.9287
    assert high_50 == 1.0


def test_similar_cohort_summary_generation():
    fake_ml_state = {
        "oof_predictions": {
            "Cath": {
                "probs": [0.1 * i for i in range(10)] * 5,  # 50 scores
                "labels": [0 if i < 5 else 1 for i in range(10)] * 5,
            }
        }
    }
    summary = compute_similar_cohort_summary(
        score=0.85, tgt="Cath", ml_state=fake_ml_state, k_neighbors=20
    )
    assert summary is not None
    assert summary["n"] == 20
    assert 0.0 <= summary["observed_rate"] <= 1.0
    assert len(summary["wilson_interval"]) == 2
    assert "Among 20 similar-scoring patients" in summary["summary"]
