"""
Tests for Feature Schema and Clinical Domain Coverage.
Asserts that all 54 features belong to exactly one of the 5 canonical physiological domains,
that no feature is unmapped, and that domain feature membership is strictly disjoint and exhaustive.
"""

import json
from pathlib import Path
import pytest

SCHEMA_PATH = Path("ml/schema.json")


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_all_54_features_present(schema):
    assert len(schema) == 54, f"Expected exactly 54 features, found {len(schema)}"


def test_all_features_have_valid_groups_and_domains(schema):
    valid_groups = {"demographic", "symptoms-exam", "lab", "ECG", "echo"}
    valid_domains = {"demographics", "symptoms_exam", "labs", "ecg", "echo"}

    for feat_name, meta in schema.items():
        grp = meta.get("group")
        dom = meta.get("domain")

        assert grp in valid_groups, f"Feature '{feat_name}' has invalid group: {grp}"
        assert dom in valid_domains, f"Feature '{feat_name}' has invalid domain: {dom}"

        # Required clinical metadata
        assert "label" in meta and len(meta["label"]) > 0, f"Missing label for {feat_name}"
        assert "encoding" in meta, f"Missing encoding for {feat_name}"
        if meta["encoding"] == "numeric":
            assert "min" in meta and "max" in meta, f"Missing min/max for numeric feature {feat_name}"
            assert meta["min"] <= meta["max"], f"min > max for {feat_name}"


def test_domain_partition_is_disjoint_and_exhaustive(schema):
    domain_counts = {}
    for feat_name, meta in schema.items():
        dom = meta["domain"]
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    expected_domains = {"demographics", "symptoms_exam", "labs", "ecg", "echo"}
    assert set(domain_counts.keys()) == expected_domains

    # Expected exact counts per domain
    assert domain_counts["demographics"] == 5, f"Expected 5 demographic features, got {domain_counts['demographics']}"
    assert domain_counts["symptoms_exam"] == 25, f"Expected 25 symptoms/exam features, got {domain_counts['symptoms_exam']}"
    assert domain_counts["labs"] == 14, f"Expected 14 lab features, got {domain_counts['labs']}"
    assert domain_counts["ecg"] == 7, f"Expected 7 ECG features, got {domain_counts['ecg']}"
    assert domain_counts["echo"] == 3, f"Expected 3 Echo features, got {domain_counts['echo']}"

    assert sum(domain_counts.values()) == 54
