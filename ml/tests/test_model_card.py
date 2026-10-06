import json
from pathlib import Path

MODEL_CARD_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model_card.json"


def test_model_card_provenance_and_notes():
    """Verify model card provenance does not mention Colab and contains metrics notes."""
    assert MODEL_CARD_PATH.exists(), f"Missing {MODEL_CARD_PATH}"

    with open(MODEL_CARD_PATH, "r", encoding="utf-8") as f:
        card = json.load(f)

    # 1. No stale Colab references
    assert "Colab" not in card["environment"]["provenance"], "Stale Colab reference in provenance"

    # 2. Aggregation and metrics notes present
    assert "aggregation_and_metrics_notes" in card, "Missing aggregation_and_metrics_notes block"
    notes = card["aggregation_and_metrics_notes"]
    assert "pr_auc_definition" in notes
    assert "vessel_target_invariance" in notes
    assert "nested_cv_evaluation" in notes
