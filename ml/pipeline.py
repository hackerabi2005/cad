"""
ML Pipeline definition for CAD and vessel-specific stenosis prediction.
Enforces zero target leakage, pipeline-contained preprocessing, and standard interfaces.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = ["Cath", "LAD", "LCX", "RCA"]
DROP_COLS = ["Cath", "LAD", "LCX", "RCA", "Exertional CP"]

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"


def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_groups(schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Separates numeric/ordinal features from categorical/binary string features."""
    numeric_features = []
    categorical_features = []

    for name, meta in schema.items():
        if meta["encoding"] in ("numeric", "ordinal"):
            numeric_features.append(name)
        elif meta["encoding"] in ("binary", "nominal"):
            categorical_features.append(name)
        else:
            categorical_features.append(name)

    return numeric_features, categorical_features


class FeatureNameTracker(BaseEstimator, TransformerMixin):
    """Tracks column names post-transformation and maps one-hot dummy columns back to parent features."""
    def __init__(self, preprocessor: ColumnTransformer):
        self.preprocessor = preprocessor
        self.transformed_feature_names_: List[str] = []
        self.feature_to_parent_: Dict[str, str] = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def build_preprocessor(schema: Dict[str, Any]) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_groups(schema)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_pipeline(estimator: Any, schema: Dict[str, Any] | None = None) -> Pipeline:
    """Builds a complete ML pipeline containing preprocessing and the given classifier."""
    if schema is None:
        schema = load_schema()

    preprocessor = build_preprocessor(schema)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    return pipeline


def load_raw_dataset(excel_path: str | Path, align_row_93: bool = True) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Loads raw Excel dataset, strictly strips target and constant columns from X,
    and returns X and a dictionary of binary target series {target_name: y_series}.
    """
    df = pd.read_excel(excel_path)

    # Clean target columns into binary 0/1
    # LAD, LCX, RCA: 'Stenotic' -> 1, 'Normal' -> 0
    # Cath: 'CAD' -> 1, 'Normal' -> 0
    y_dict = {}
    for target in ["LAD", "LCX", "RCA"]:
        y_dict[target] = (df[target].astype(str).str.strip().str.lower() == "stenotic").astype(int)

    cath_series = (df["Cath"].astype(str).str.strip().str.lower() == "cad").astype(int)

    vessels_or = y_dict["LAD"] | y_dict["LCX"] | y_dict["RCA"]
    mismatch_mask = (cath_series != vessels_or)
    mismatch_indices = list(df.index[mismatch_mask])

    if align_row_93:
        # Align Cath to match dataset's definition (CAD = >= 1 stenotic vessel)
        assert len(mismatch_indices) == 1, (
            f"Expected exactly 1 raw mismatch row between Cath and OR(vessels), found {len(mismatch_indices)}: {mismatch_indices}"
        )
        cath_series = cath_series | vessels_or

    y_dict["Cath"] = cath_series

    # Extract X, ensuring strict exclusion of all targets and constant column
    X = df.drop(columns=[col for col in DROP_COLS if col in df.columns], errors="ignore").copy()

    # Cast string columns cleanly
    schema = load_schema()
    for col in X.columns:
        if col in schema and schema[col]["encoding"] == "binary" and schema[col].get("categories") == [0, 1]:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)

    return X, y_dict
