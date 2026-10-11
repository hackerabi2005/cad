import numpy as np
import pytest
from api.main import load_artifacts, ml_state
from ml.service import predict_patient
from ml.pipeline import DEFAULT_DATA_PATH, load_raw_dataset


@pytest.fixture(scope="module")
def setup_ml():
    load_artifacts()
    return ml_state


def test_clinical_direction_probes(setup_ml):
    """
    Direction probes check that increasing key cardiovascular risk factors
    (systolic BP, fasting blood sugar FBS, serum LDL) monotonically increases
    or maintains predicted risk, verifying physiological plausibility.
    Tested across the full 303-patient cohort.
    """
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)
    pipe = setup_ml["pipelines"]["Cath"]

    base_probs = pipe.predict_proba(X)[:, 1]

    # Probe 1: Systolic BP elevation (+20 mmHg)
    X_bp = X.copy()
    X_bp["BP"] = X_bp["BP"].astype(float) + 20
    p_bp = pipe.predict_proba(X_bp)[:, 1]
    monotone_bp = np.sum(p_bp >= base_probs - 1e-4)
    assert monotone_bp == len(X), f"BP elevation violated monotonicity on {len(X) - monotone_bp} rows"

    # Probe 2: Fasting Blood Sugar elevation (+30 mg/dL)
    X_fbs = X.copy()
    X_fbs["FBS"] = X_fbs["FBS"].astype(float) + 30
    p_fbs = pipe.predict_proba(X_fbs)[:, 1]
    monotone_fbs = np.sum(p_fbs >= base_probs - 1e-4)
    assert monotone_fbs == len(X), f"FBS elevation violated monotonicity on {len(X) - monotone_fbs} rows"

    # Probe 3: Serum LDL elevation (+30 mg/dL)
    X_ldl = X.copy()
    X_ldl["LDL"] = X_ldl["LDL"].astype(float) + 30
    p_ldl = pipe.predict_proba(X_ldl)[:, 1]
    monotone_ldl = np.sum(p_ldl >= base_probs - 1e-4)
    assert monotone_ldl == len(X), f"LDL elevation violated monotonicity on {len(X) - monotone_ldl} rows"

    # Probe 4: Smoking status (Non-smoker -> Active Current Smoker) on row 0
    base = X.iloc[0].to_dict()
    p_base_0 = predict_patient(base, setup_ml)["cad"]["prob"]
    mod_smoke = dict(base, **{"Current Smoker": 1, "EX-Smoker": 0})
    p_smoke = predict_patient(mod_smoke, setup_ml)["cad"]["prob"]
    assert p_smoke >= p_base_0 - 1e-6, f"Active smoking initiation lowered risk: {p_smoke} < {p_base_0}"


def test_clinical_rules_vs_dataset_agreement():
    """
    Empirical agreement test verifying clinical flag definitions across all 303 cohort rows:
    1. BMI > 25 agrees with Obesity == 'Y' on >99% of rows (302/303), confirming dataset definition.
    2. BP >= 140 vs HTN == 1 has ~74% agreement (225/303), proving HTN is a recorded clinical diagnosis.
    3. (LDL >= 130 or TG >= 150) vs DLP == 'Y' has ~53% agreement (161/303), proving DLP is a diagnosed
       condition (often medicated with statins) rather than an acute lab threshold.
    """
    X, _ = load_raw_dataset(DEFAULT_DATA_PATH)

    # 1. Obesity rule (dataset definition: BMI > 25)
    bmi_obese = (X["BMI"] > 25.0)
    obesity_col = (X["Obesity"] == "Y")
    obesity_agreement = (bmi_obese == obesity_col).mean()
    assert obesity_agreement > 0.99, f"BMI > 25 agreement with Obesity was {obesity_agreement:.3f}, expected > 0.99"

    # 2. HTN diagnosis vs acute resting BP >= 140
    bp_140 = (X["BP"] >= 140)
    htn_col = (X["HTN"] == 1)
    htn_agreement = (bp_140 == htn_col).mean()
    # HTN is a diagnosed disease history; many hypertensive patients have normal BP on medications
    assert 0.70 <= htn_agreement <= 0.80, f"HTN agreement was {htn_agreement:.3f}, expected ~0.74"

    # 3. Dyslipidemia diagnosis vs acute lipid panel
    dlp_rule = ((X["LDL"] >= 130) | (X["TG"] >= 150))
    dlp_col = (X["DLP"] == "Y")
    dlp_agreement = (dlp_rule == dlp_col).mean()
    # DLP is a recorded clinical diagnosis; statin therapy frequently normalizes LDL below 130
    assert 0.45 <= dlp_agreement <= 0.60, f"DLP agreement was {dlp_agreement:.3f}, expected ~0.53"

