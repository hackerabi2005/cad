"""
Determinism test for XGBoost models (B2).
Verifies that with subsample=1.0 and colsample_bytree=1.0, XGBoost predictions
are bit-identical across different seeds, eliminating Linux/Windows cross-platform drift.
"""

from pathlib import Path
import numpy as np
import pytest
from xgboost import XGBClassifier
from ml.pipeline import load_raw_dataset, build_pipeline, DEFAULT_DATA_PATH


def test_xgboost_rng_free_determinism():
    X, y = load_raw_dataset(DEFAULT_DATA_PATH)

    def fit_and_predict(seed: int):
        model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.08,
            subsample=1.0,
            colsample_bytree=1.0,
            n_jobs=1,
            random_state=seed,
            eval_metric="logloss",
        )
        pipe = build_pipeline(model)
        pipe.fit(X, y["LCX"])
        return pipe.predict_proba(X)[:, 1]

    pred_seed1 = fit_and_predict(42)
    pred_seed2 = fit_and_predict(999)

    max_diff = float(np.max(np.abs(pred_seed1 - pred_seed2)))
    assert max_diff == 0.0, f"XGBoost RNG-free predictions differed by {max_diff} across seeds!"
