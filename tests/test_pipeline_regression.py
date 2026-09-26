"""
Regression safety test suite to ensure all existing MLForge modules
continue to function flawlessly alongside the new Data Quality & ML Readiness Audit.
"""

import io
import os
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification

from modules.file_loader import load_file
from modules.profiling import get_basic_info, get_column_summary, get_numeric_stats, get_categorical_stats
from modules.missing_handler import get_missing_summary, fill_missing_values, suggest_missing_strategy
from modules.duplicate_handler import remove_duplicates
from modules.clustering import run_all_clustering
from modules.ai_recommender import generate_ai_report, recommend_clustering, recommend_model_export
from modules.exporter import export_data
from modules.model_export.export_manager import export_model
from modules.data_audit import run_data_audit


@pytest.fixture
def sample_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    return df


def test_profiling_module(sample_data):
    info = get_basic_info(sample_data)
    assert info["rows"] == 150
    assert info["columns"] == 5

    summary = get_column_summary(sample_data)
    assert len(summary) == 5

    num_stats = get_numeric_stats(sample_data)
    assert not num_stats.empty


def test_missing_handler_module():
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["a", np.nan, "b", "b"]
    })
    miss_sum = get_missing_summary(df)
    assert len(miss_sum) == 2
    
    suggested = suggest_missing_strategy(df)
    assert "num" in suggested
    
    filled = fill_missing_values(df, {"num": "mean", "cat": "mode"})
    assert filled["num"].isna().sum() == 0
    assert filled["cat"].isna().sum() == 0


def test_duplicate_handler_module():
    df = pd.DataFrame({
        "a": [1, 2, 1],
        "b": ["x", "y", "x"]
    })
    cleaned, stats = remove_duplicates(df, subset=None, keep="first")
    assert stats["removed"] == 1
    assert len(cleaned) == 2


def test_ai_recommender_module(sample_data):
    report = generate_ai_report(sample_data, target_col="target")
    assert report["problem_type"] == "classification"
    assert len(report["recommended_models"]) > 0
    assert "dataset" in report


def test_clustering_module(sample_data):
    features = sample_data.drop(columns=["target"])
    res = run_all_clustering(features)
    assert "KMeans" in res


def test_data_export_module(sample_data):
    content, mime = export_data(sample_data, "csv")
    assert isinstance(content, (str, bytes, bytearray))
    assert mime == "text/csv"
    assert len(content) > 0


def test_model_export_module():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.randn(20, 3)
    y = np.random.choice([0, 1], size=20)
    rf.fit(X, y)

    # Test joblib export
    data_j, mime_j = export_model(rf, "joblib")
    assert data_j is not None
    assert isinstance(data_j, bytes)
    assert len(data_j) > 0
    assert "octet-stream" in mime_j

    # Test pickle export
    data_p, mime_p = export_model(rf, "pickle")
    assert data_p is not None
    assert isinstance(data_p, bytes)
    assert len(data_p) > 0

    # Test ONNX export with sample data
    data_o, mime_o = export_model(rf, "onnx", X[:5])
    assert data_o is not None
    assert isinstance(data_o, bytes)
    assert len(data_o) > 0


def test_audit_integration_with_pipeline_flow(sample_data):
    # Step 1: Raw upload audit
    audit1 = run_data_audit(sample_data)
    assert audit1.score >= 80

    # Step 2: Inject missing values and verify audit reflects them
    dirty_df = sample_data.copy()
    dirty_df.iloc[0:30, 0] = np.nan
    audit2 = run_data_audit(dirty_df)
    assert audit2.dataset_stats["missing_cells"] == 30
    assert audit2.score < audit1.score

    # Step 3: Clean missing values and verify audit score recovers
    filled_df = fill_missing_values(dirty_df, {"sepal length (cm)": "median"})
    audit3 = run_data_audit(filled_df)
    assert audit3.dataset_stats["missing_cells"] == 0
    assert audit3.score > audit2.score
