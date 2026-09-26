"""
Comprehensive test suite for modules/data_audit.py
Covers all 15 mandatory test scenarios and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from modules.data_audit import DataAuditor, run_data_audit, AuditResult


# =====================================================================
# 1. Clean dataset
# =====================================================================
def test_clean_dataset():
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_a": np.random.normal(loc=10, scale=2, size=150),
        "feature_b": np.random.uniform(0, 100, size=150),
        "feature_c": np.random.choice(["cat", "dog", "bird"], size=150),
        "target": np.random.choice([0, 1], size=150)
    })
    result = run_data_audit(df, target_col="target")
    assert isinstance(result, AuditResult)
    assert result.score >= 80
    assert result.dataset_stats["missing_cells"] == 0
    assert result.dataset_stats["duplicate_rows"] == 0
    assert result.severity in ("Low", "Info")


# =====================================================================
# 2. Dataset with missing values
# =====================================================================
def test_dataset_with_missing_values():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,
        "num2": [np.nan] * 50 + [1.0] * 50,  # 50% missing
        "cat1": ["a", "b", None, "a", "b"] * 20
    })
    result = run_data_audit(df)
    assert result.dataset_stats["missing_cells"] > 0
    assert result.dataset_stats["missing_pct"] > 0
    assert "num2" in result.classified_columns["high_missing"]
    # Check that missing data issue was detected
    missing_issues = [i for i in result.detected_issues if i.category == "Missing Data"]
    assert len(missing_issues) > 0
    assert any(d.category == "Missing Data" for d in result.score_breakdown)


# =====================================================================
# 3. Dataset with duplicates
# =====================================================================
def test_dataset_with_duplicates():
    df = pd.DataFrame({
        "x": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4] * 10,
        "y": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "d"] * 10
    })
    result = run_data_audit(df)
    assert result.dataset_stats["duplicate_rows"] > 0
    assert result.dataset_stats["duplicate_pct"] > 0
    dup_issues = [i for i in result.detected_issues if i.category == "Duplicates"]
    assert len(dup_issues) > 0
    assert any("duplicate" in rec.lower() for rec in result.recommendations)


# =====================================================================
# 4. Dataset with constant column
# =====================================================================
def test_dataset_with_constant_column():
    df = pd.DataFrame({
        "const_num": [42.0] * 100,
        "const_str": ["FIXED"] * 100,
        "normal": np.random.randn(100)
    })
    result = run_data_audit(df)
    assert "const_num" in result.classified_columns["constant"]
    assert "const_str" in result.classified_columns["constant"]
    const_issues = [i for i in result.detected_issues if i.column in ("const_num", "const_str")]
    assert len(const_issues) >= 2


# =====================================================================
# 5. Dataset with near-constant column
# =====================================================================
def test_dataset_with_near_constant_column():
    df = pd.DataFrame({
        "near_const": ["yes"] * 97 + ["no"] * 3,  # 97% dominant value
        "normal": np.arange(100)
    })
    result = run_data_audit(df)
    assert "near_const" in result.classified_columns["near_constant"]
    near_const_issues = [i for i in result.detected_issues if i.column == "near_const"]
    assert len(near_const_issues) > 0


# =====================================================================
# 6. Dataset with ID-like column
# =====================================================================
def test_dataset_with_id_like_column():
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(100)],
        "id": range(1, 101),
        "amount": np.random.uniform(10, 500, size=100)
    })
    result = run_data_audit(df)
    assert "customer_id" in result.classified_columns["id_like"]
    assert "id" in result.classified_columns["id_like"]
    id_issues = [i for i in result.detected_issues if i.column in ("customer_id", "id")]
    assert len(id_issues) >= 1
    # Check that recommendation mentions excluding ID
    assert any("feature" in i.recommendation.lower() for i in id_issues)


# =====================================================================
# 7. Dataset with categorical imbalance
# =====================================================================
def test_dataset_with_categorical_imbalance():
    df = pd.DataFrame({
        "feat1": np.random.randn(100),
        "label": [0] * 96 + [1] * 4  # 4% minority class (severe imbalance)
    })
    result = run_data_audit(df, target_col="label")
    assert result.target_analysis is not None
    assert result.target_analysis.is_severely_imbalanced is True
    assert result.target_analysis.minority_pct == 4.0
    imbalance_issues = [i for i in result.detected_issues if i.category == "Target" and "imbalance" in i.message.lower()]
    assert len(imbalance_issues) > 0


# =====================================================================
# 8. Dataset with numeric outliers
# =====================================================================
def test_dataset_with_numeric_outliers():
    # Regular values around 10, with a few extreme values
    vals = [10.0, 10.5, 9.8, 10.2, 10.1, 9.9, 10.3, 10.0] * 12
    vals[0] = 9999.0
    vals[1] = -5000.0
    df = pd.DataFrame({"outlier_feat": vals, "feat_b": np.random.randn(len(vals))})
    result = run_data_audit(df)
    assert "outlier_feat" in result.classified_columns["outliers"]
    assert result.outlier_summary["total_outliers"] >= 2
    assert result.column_profiles["outlier_feat"].outlier_count >= 2


# =====================================================================
# 9. Dataset with date column
# =====================================================================
def test_dataset_with_date_column():
    df = pd.DataFrame({
        "dt_real": pd.date_range("2024-01-01", periods=50),
        "dt_string": [f"2024-05-{i:02d}" for i in range(1, 29)] + [f"2024-06-{i:02d}" for i in range(1, 23)],
        "val": np.random.randn(50)
    })
    result = run_data_audit(df)
    assert "dt_real" in result.classified_columns["datetime"]
    assert "dt_string" in result.classified_columns["datetime"]
    assert result.dataset_stats["datetime_cols_count"] == 2


# =====================================================================
# 10. Dataset containing infinite values
# =====================================================================
def test_dataset_with_infinite_values():
    df = pd.DataFrame({
        "feat_inf": [1.0, 2.0, np.inf, 4.0, -np.inf] * 10,
        "normal": np.arange(50)
    })
    result = run_data_audit(df)
    assert "feat_inf" in result.classified_columns["infinite"]
    inf_issues = [i for i in result.detected_issues if "infinite" in i.message.lower()]
    assert len(inf_issues) > 0
    assert any(d.category == "Infinite Values" for d in result.score_breakdown)


# =====================================================================
# 11. Empty dataset
# =====================================================================
def test_empty_dataset():
    df_empty = pd.DataFrame()
    result = run_data_audit(df_empty)
    assert result.score == 0
    assert result.severity == "Critical"
    assert result.grade == "Critical (Unusable)"

    df_zero_rows = pd.DataFrame({"a": [], "b": []})
    result2 = run_data_audit(df_zero_rows)
    assert result2.score == 0
    assert result2.severity == "Critical"


# =====================================================================
# 12. Very small dataset
# =====================================================================
def test_very_small_dataset():
    df_small = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50]
    })
    result = run_data_audit(df_small)
    assert result.dataset_stats["rows"] == 5
    shape_issues = [i for i in result.detected_issues if i.category == "Dataset Shape"]
    assert len(shape_issues) > 0
    assert any(d.category == "Dataset Size" for d in result.score_breakdown)


# =====================================================================
# 13. Dataset without target
# =====================================================================
def test_dataset_without_target():
    df = pd.DataFrame({
        "col1": np.random.randn(80),
        "col2": np.random.uniform(0, 10, size=80),
        "col3": ["type_a", "type_b"] * 40
    })
    result = run_data_audit(df, target_col=None)
    assert result.target_analysis is None
    assert result.score > 0
    assert result.score <= 100


# =====================================================================
# 14. Classification target
# =====================================================================
def test_classification_target():
    df = pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100),
        "class_label": ["setosa", "versicolor", "virginica"] * 33 + ["setosa"]
    })
    result = run_data_audit(df, target_col="class_label")
    assert result.target_analysis is not None
    assert result.target_analysis.task_type == "classification"
    assert result.target_analysis.n_classes == 3
    assert "setosa" in result.target_analysis.class_distribution
    assert result.target_analysis.is_severely_imbalanced is False


# =====================================================================
# 15. Regression target
# =====================================================================
def test_regression_target():
    df = pd.DataFrame({
        "f1": np.random.randn(120),
        "price": np.random.exponential(scale=50000, size=120)
    })
    result = run_data_audit(df, target_col="price")
    assert result.target_analysis is not None
    assert result.target_analysis.task_type == "regression"
    assert result.target_analysis.regression_stats is not None
    assert "mean" in result.target_analysis.regression_stats
    assert "std" in result.target_analysis.regression_stats


# =====================================================================
# 16. Edge Cases & Leakage
# =====================================================================
def test_target_leakage_detection():
    target = np.random.randn(100)
    df = pd.DataFrame({
        "exact_leak": target,
        "correlated_leak": target * 1.0001,
        "legit_feature": np.random.randn(100),
        "target_col": target
    })
    result = run_data_audit(df, target_col="target_col")
    assert result.target_analysis is not None
    assert len(result.target_analysis.leakage_suspects) >= 1
    leak_issues = [i for i in result.detected_issues if i.category == "Data Leakage"]
    assert len(leak_issues) >= 1
    assert any(issue.severity == "Critical" for issue in leak_issues)


def test_duplicate_columns_detection():
    df = pd.DataFrame({
        "original": [10, 20, 30, 40, 50] * 10,
        "exact_copy": [10, 20, 30, 40, 50] * 10,
        "other": range(50)
    })
    result = run_data_audit(df)
    dup_col_issues = [i for i in result.detected_issues if i.category == "Duplicates" and isinstance(i.column, list)]
    assert len(dup_col_issues) > 0


def test_all_null_target_column():
    df = pd.DataFrame({
        "f1": range(50),
        "target": [np.nan] * 50
    })
    result = run_data_audit(df, target_col="target")
    assert result.target_analysis is not None
    assert result.target_analysis.valid_count == 0
    target_issues = [i for i in result.detected_issues if i.severity == "Critical" and i.category == "Target"]
    assert len(target_issues) > 0


def test_high_cardinality_categorical():
    df = pd.DataFrame({
        "city_names": [f"City_{i}" for i in range(70)],
        "val": range(70)
    })
    result = run_data_audit(df)
    assert "city_names" in result.classified_columns["high_cardinality"]
    card_issues = [i for i in result.detected_issues if "cardinality" in i.message.lower()]
    assert len(card_issues) > 0


def test_score_breakdown_and_bounds():
    df = pd.DataFrame({
        "id": range(100),
        "empty_col": [np.nan] * 100,
        "const_col": [9] * 100,
        "inf_col": [1.0, np.inf] * 50,
        "target": [1] * 99 + [0]  # 1% minority
    })
    result = run_data_audit(df, target_col="target")
    assert 0 <= result.score <= 100
    assert len(result.score_breakdown) > 0
    # Every deduction must have positive penalty and valid reason
    for d in result.score_breakdown:
        assert d.penalty > 0
        assert len(d.reason) > 0
