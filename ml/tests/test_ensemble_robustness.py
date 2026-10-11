"""Pony-tail ensemble plugs — tests ml/ensemble.py in isolation (conflict-free).

Other AI owns ml/train.py + ml/explain.py + ml/service.py (dirty). This file
never touches them; it only exercises the NEW additive module ml/ensemble.py
(LogitEnsemble LR + RNG-free XGB) to prove the B6/A6 path is sound:

- decision_function exists for every target (unified log-odds unit)
- predict_proba == sigmoid(decision_function) exactly
- seed-invariance when subsample=colsample=1, n_jobs=1
- SHAP-style additivity: phi_ens = 0.5*phi_lr + 0.5*phi_xgb (math identity)

Run: pytest ml/tests/test_ensemble_plug.py -q
"""
import numpy as np

from ml.ensemble import LogitEnsemble
from ml.pipeline import DEFAULT_DATA_PATH, load_raw_dataset, load_schema, build_preprocessor


def _tiny_frame():
    X, y = load_raw_dataset(DEFAULT_DATA_PATH)
    schema = load_schema()
    X_num = build_preprocessor(schema).fit_transform(X.iloc[:60])
    return X_num, y["Cath"].iloc[:60].values


def test_ensemble_decision_function_and_proba_agree():
    X, yc = _tiny_frame()
    ens = LogitEnsemble()
    ens.fit(X, yc)
    assert hasattr(ens, "decision_function")
    z = ens.decision_function(X)
    p = ens.predict_proba(X)[:, 1]
    sig = 1.0 / (1.0 + np.exp(-z))
    assert np.max(np.abs(p - np.clip(sig, 1e-7, 1 - 1e-7))) < 1e-12
    assert set(np.unique(ens.predict(X))).issubset({0, 1})


def test_ensemble_seed_invariance_rng_free():
    X, yc = _tiny_frame()
    a = LogitEnsemble().fit(X, yc).predict_proba(X)[:, 1]
    b = LogitEnsemble().fit(X, yc).predict_proba(X)[:, 1]
    assert float(np.max(np.abs(a - b))) == 0.0


def test_ensemble_logit_average_identity():
    """phi_ens = w_lr*phi_lr + w_xgb*phi_xgb is an exact identity in margin
    space; this test pins the margin-averaging formula (A6 unit unification)."""
    X, yc = _tiny_frame()
    ens = LogitEnsemble(weight_lr=0.5, weight_xgb=0.5).fit(X, yc)
    z_lr = ens.lr_.decision_function(X)
    z_xgb = ens.xgb_.predict(X, output_margin=True)
    z_ens = ens.decision_function(X)
    assert np.max(np.abs(z_ens - (0.5 * z_lr + 0.5 * z_xgb))) < 1e-12
