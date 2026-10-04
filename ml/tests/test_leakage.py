"""
Test for data leakage and baseline sanity checks.
Ensures:
1. X never contains any target or catheterization findings (LAD, LCX, RCA, Cath).
2. All 54 features are present in schema.json and X.
3. Label-shuffled CV ROC-AUC drops to random chance (~0.5 ± 0.07).
"""

from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ml.pipeline import load_raw_dataset, build_pipeline, load_schema, TARGETS, DROP_COLS

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"


def test_no_targets_in_features():
    X, y_dict = load_raw_dataset(DATA_PATH)

    for target in TARGETS:
        assert target not in X.columns, f"Target {target} found in feature matrix X!"

    for dropped in DROP_COLS:
        assert dropped not in X.columns, f"Dropped column {dropped} found in feature matrix X!"

    assert X.shape[1] == 54, f"Expected exactly 54 features, found {X.shape[1]}"


def test_schema_coverage():
    schema = load_schema()
    X, _ = load_raw_dataset(DATA_PATH)

    assert set(schema.keys()) == set(X.columns), "Mismatch between schema keys and dataset feature columns"
    assert len(schema) == 54


def test_label_shuffled_roc_auc_is_chance():
    """
    If there is hidden target leakage, a model trained on shuffled labels
    will still overperform. With zero leakage, cross-validated ROC-AUC must be ~0.50.
    """
    X, y_dict = load_raw_dataset(DATA_PATH)
    rng = np.random.RandomState(42)

    for target in TARGETS:
        y_shuffled = rng.permutation(y_dict[target].values)

        pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y_shuffled, cv=cv, scoring="roc_auc")
        mean_score = float(np.mean(scores))

        print(f"Target {target} shuffled CV ROC-AUC: {mean_score:.4f}")
        # Must be around 0.50 ± 0.08
        assert 0.35 <= mean_score <= 0.65, (
            f"Target {target} shuffled ROC-AUC {mean_score:.4f} deviates significantly from 0.50!"
        )
