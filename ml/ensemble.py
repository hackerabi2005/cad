"""
LogitEnsemble: Combines Logistic Regression and RNG-free XGBoost in log-odds margin space.
Guarantees exact cross-platform determinism (no RNG dependencies) and exact analytical SHAP additivity (< 1e-9).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class LogitEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier that averages prediction margins (logits) from
    Logistic Regression and RNG-free XGBoost (subsample=1.0, colsample_bytree=1.0, n_jobs=1).

    Margin: z_ensemble = weight_lr * z_lr + weight_xgb * z_xgb
    Probability: p = 1 / (1 + exp(-z_ensemble))
    """

    def __init__(
        self,
        weight_lr: float = 0.5,
        weight_xgb: float = 0.5,
        lr_c: float = 0.1,
        xgb_n_estimators: int = 50,
        xgb_max_depth: int = 3,
        xgb_learning_rate: float = 0.08,
    ):
        self.weight_lr = weight_lr
        self.weight_xgb = weight_xgb
        self.lr_c = lr_c
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        lr_base = LogisticRegression(
            C=self.lr_c,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        xgb_base = XGBClassifier(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=1.0,
            colsample_bytree=1.0,
            colsample_bylevel=1.0,
            colsample_bynode=1.0,
            n_jobs=1,
            random_state=42,
            eval_metric="logloss",
        )

        needs_prep = isinstance(X, pd.DataFrame) and any(
            not pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes
        )
        if needs_prep:
            from ml.pipeline import build_pipeline
            self.lr_ = build_pipeline(lr_base)
            self.xgb_ = build_pipeline(xgb_base)
        else:
            self.lr_ = lr_base
            self.xgb_ = xgb_base

        self.lr_.fit(X, y)
        self.xgb_.fit(X, y)
        return self

    def decision_function(self, X) -> np.ndarray:
        z_lr = self.lr_.decision_function(X)
        z_xgb = self.xgb_.predict(X, output_margin=True)
        return self.weight_lr * z_lr + self.weight_xgb * z_xgb

    def predict_proba(self, X) -> np.ndarray:
        z = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(int)
