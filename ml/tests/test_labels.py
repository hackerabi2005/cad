import hashlib
import pandas as pd
import pytest
from ml.pipeline import DEFAULT_DATA_PATH, load_raw_dataset

RAW_DATASET_SHA256 = "739343245c2ba578b541370217531750d8e936022f928b83e0d91756caa3ff0b"


def test_raw_file_hash_unchanged():
    """Verify that the raw Excel file on disk remains completely untouched."""
    with open(DEFAULT_DATA_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == RAW_DATASET_SHA256, f"Raw data file hash mismatch: {file_hash} != {RAW_DATASET_SHA256}"


def test_raw_mismatch_set_is_exactly_row_93():
    """Verify that the raw data contains exactly one mismatch between Cath and OR(LAD, LCX, RCA)."""
    df = pd.read_excel(DEFAULT_DATA_PATH)
    assert len(df) == 303

    lad = (df["LAD"].astype(str).str.strip().str.lower() == "stenotic").astype(int)
    lcx = (df["LCX"].astype(str).str.strip().str.lower() == "stenotic").astype(int)
    rca = (df["RCA"].astype(str).str.strip().str.lower() == "stenotic").astype(int)
    cath = (df["Cath"].astype(str).str.strip().str.lower() == "cad").astype(int)

    mismatches = df[cath != (lad | lcx | rca)]
    assert list(mismatches.index) == [93], f"Expected mismatch at index [93], got {list(mismatches.index)}"

    row = mismatches.iloc[0]
    assert row["Age"] == 65
    assert str(row["Sex"]).strip().lower() == "male"
    assert row["Weight"] == 73
    assert row["Length"] == 165
    assert str(row["Cath"]).strip().lower() == "normal"
    assert str(row["LAD"]).strip().lower() == "stenotic"
    assert str(row["LCX"]).strip().lower() == "normal"
    assert str(row["RCA"]).strip().lower() == "normal"


def test_aligned_cath_equals_or_vessels_all_rows():
    """Verify that aligned Cath exactly equals OR(LAD, LCX, RCA) across all 303 rows."""
    X, y_dict = load_raw_dataset(DEFAULT_DATA_PATH, align_row_93=True)
    assert len(X) == 303

    vessels_or = y_dict["LAD"] | y_dict["LCX"] | y_dict["RCA"]
    diff = y_dict["Cath"] != vessels_or
    assert diff.sum() == 0, f"Found {diff.sum()} mismatches after alignment"
    assert y_dict["Cath"].sum() == 217
    assert (y_dict["Cath"] == 0).sum() == 86


def test_unaligned_cath_matches_raw_counts():
    """Verify unaligned Cath preserves the raw 216 / 87 distribution."""
    X, y_dict = load_raw_dataset(DEFAULT_DATA_PATH, align_row_93=False)
    assert y_dict["Cath"].sum() == 216
    assert (y_dict["Cath"] == 0).sum() == 87


def test_row_93_sensitivity_analysis():
    """Verify that training on unaligned (216/87) vs aligned (217/86) shifts CV ROC-AUC by < 0.01."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from ml.pipeline import DEFAULT_DATA_PATH, load_raw_dataset, build_preprocessor, load_schema

    schema = load_schema()
    preprocessor = build_preprocessor(schema)
    clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X_raw, y_raw = load_raw_dataset(DEFAULT_DATA_PATH, align_row_93=False)
    score_raw = cross_val_score(pipe, X_raw, y_raw["Cath"], cv=cv, scoring="roc_auc").mean()

    X_aln, y_aln = load_raw_dataset(DEFAULT_DATA_PATH, align_row_93=True)
    score_aln = cross_val_score(pipe, X_aln, y_aln["Cath"], cv=cv, scoring="roc_auc").mean()

    delta = abs(score_aln - score_raw)
    assert delta < 0.01, f"Row 93 alignment caused unexpected large shift: {delta}"
