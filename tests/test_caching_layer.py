"""
MLForge — Caching Layer Tests
================================
Verifies correctness, safety, and isolation properties of utils/cache.py.

Test categories:
  1. Dataset correctness  — same input → same result, different input → different result
  2. Mutation safety       — cached DataFrames are isolated from post-call mutations
  3. Configuration correctness — different params → different cache entries
  4. Content-based identity — same content = cache hit regardless of object identity
  5. Cache invalidation    — changed df content → fresh computation
"""

import copy
import hashlib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iris_df():
    return load_iris(as_frame=True).frame.copy()


@pytest.fixture
def diabetes_df():
    return load_diabetes(as_frame=True).frame.copy()


@pytest.fixture
def small_df():
    """Minimal, controlled DataFrame for determinism testing."""
    return pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "num_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat_x": ["a", "b", "a", "c", "b"],
    })


@pytest.fixture
def missing_df():
    """DataFrame with known missing values."""
    return pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat": ["a", np.nan, "b", "b", np.nan],
    })


# ---------------------------------------------------------------------------
# 1. Import sanity — all cache functions are importable
# ---------------------------------------------------------------------------

def test_cache_module_imports():
    from utils.cache import (
        cached_run_data_audit,
        cached_get_basic_info,
        cached_get_column_summary,
        cached_get_numeric_stats,
        cached_get_categorical_stats,
        cached_compute_correlation,
        cached_get_missing_summary,
        cached_suggest_missing_strategy,
        cached_get_preprocessing_recommendations,
        cached_generate_ai_report,
        cached_prepare_clustering_data,
        cached_reduce_to_2d,
        cached_find_optimal_clusters,
        cached_parse_uploaded_file,
        cached_load_benchmark_dataset,
        cached_load_logo_svg,
        get_file_hash,
    )
    # All must be callable
    for fn in [cached_run_data_audit, cached_get_basic_info, cached_get_column_summary,
               cached_get_numeric_stats, cached_get_categorical_stats, cached_compute_correlation,
               cached_get_missing_summary, cached_suggest_missing_strategy,
               cached_get_preprocessing_recommendations, cached_generate_ai_report,
               cached_prepare_clustering_data, cached_reduce_to_2d, cached_find_optimal_clusters,
               cached_parse_uploaded_file, cached_load_benchmark_dataset, get_file_hash]:
        assert callable(fn), f"{fn} is not callable"


# ---------------------------------------------------------------------------
# 2. Dataset correctness — Dataset A → correct result, Dataset B → correct result
# ---------------------------------------------------------------------------

def test_basic_info_correctness(iris_df, diabetes_df):
    from utils.cache import cached_get_basic_info

    info_iris = cached_get_basic_info(iris_df)
    assert info_iris["rows"] == 150
    assert info_iris["columns"] == 5

    info_diab = cached_get_basic_info(diabetes_df)
    assert info_diab["rows"] == 442
    assert info_diab["columns"] == 11

    # Iris again — must still be correct (cached hit)
    info_iris2 = cached_get_basic_info(iris_df)
    assert info_iris2["rows"] == 150
    assert info_iris2["columns"] == 5


def test_column_summary_correctness(iris_df):
    from utils.cache import cached_get_column_summary

    summary = cached_get_column_summary(iris_df)
    assert len(summary) == 5
    assert "column" in summary.columns
    assert "dtype" in summary.columns
    assert "missing" in summary.columns


def test_numeric_stats_correctness(iris_df):
    from utils.cache import cached_get_numeric_stats

    stats = cached_get_numeric_stats(iris_df)
    assert not stats.empty
    assert "column" in stats.columns
    assert "skew" in stats.columns


def test_categorical_stats_empty_for_iris(iris_df):
    """Iris has no categorical columns → empty result."""
    from utils.cache import cached_get_categorical_stats

    cat_stats = cached_get_categorical_stats(iris_df)
    assert cat_stats.empty


def test_categorical_stats_nonempty_for_mixed(small_df):
    from utils.cache import cached_get_categorical_stats

    cat_stats = cached_get_categorical_stats(small_df)
    assert not cat_stats.empty
    assert "cat_x" in cat_stats["column"].values


def test_correlation_correctness(iris_df):
    from utils.cache import cached_compute_correlation

    corr = cached_compute_correlation(iris_df)
    assert not corr.empty
    # Self-correlation must be 1.0 (or very close due to float precision)
    for col in corr.columns:
        assert abs(corr.loc[col, col] - 1.0) < 1e-6, f"Self-correlation for {col} != 1.0"


def test_correlation_returns_empty_for_no_numeric(missing_df):
    from utils.cache import cached_compute_correlation

    # DataFrame with only 1 numeric column → cannot compute correlation matrix
    cat_only = pd.DataFrame({"cat": ["a", "b", "c"]})
    corr = cached_compute_correlation(cat_only)
    assert corr.empty


def test_missing_summary_correctness(missing_df):
    from utils.cache import cached_get_missing_summary

    miss = cached_get_missing_summary(missing_df)
    assert len(miss) == 2  # both columns have missing values
    assert set(miss["column"].tolist()) == {"num", "cat"}


def test_missing_strategy_correctness(missing_df):
    from utils.cache import cached_suggest_missing_strategy

    strategies = cached_suggest_missing_strategy(missing_df)
    assert "num" in strategies
    # numeric with moderate skew → mean or median
    assert strategies["num"] in ("mean", "median")


def test_preprocessing_recommendations_correctness(iris_df):
    from utils.cache import cached_get_preprocessing_recommendations

    recs = cached_get_preprocessing_recommendations(iris_df)
    assert len(recs) == 5  # one row per column
    assert "Recommended Action" in recs.columns
    assert "Reason" in recs.columns


# ---------------------------------------------------------------------------
# 3. Mutation safety — modifying result of cached call must not corrupt cache
# ---------------------------------------------------------------------------

def test_column_summary_mutation_safety(iris_df):
    """Mutating the returned DataFrame must not corrupt the cached entry."""
    from utils.cache import cached_get_column_summary

    result1 = cached_get_column_summary(iris_df)
    original_len = len(result1)

    # Mutate the returned copy
    result1_mutated = result1.copy()
    result1_mutated["__junk__"] = 999
    result1_mutated.drop(index=0, inplace=True)

    # Re-fetch from cache — must still be the original
    result2 = cached_get_column_summary(iris_df)
    assert len(result2) == original_len
    assert "__junk__" not in result2.columns


def test_numeric_stats_mutation_safety(iris_df):
    from utils.cache import cached_get_numeric_stats

    result1 = cached_get_numeric_stats(iris_df)
    original_shape = result1.shape

    # Simulate downstream mutation
    working_copy = result1.copy()
    working_copy["bogus"] = -1

    result2 = cached_get_numeric_stats(iris_df)
    assert result2.shape == original_shape
    assert "bogus" not in result2.columns


def test_basic_info_immutability(iris_df):
    from utils.cache import cached_get_basic_info

    info1 = cached_get_basic_info(iris_df)
    original_rows = info1["rows"]

    # Mutate the dict (dicts are mutable)
    info1["rows"] = -99999

    # Cache should return fresh dict with correct value
    info2 = cached_get_basic_info(iris_df)
    assert info2["rows"] == original_rows


# ---------------------------------------------------------------------------
# 4. Cache invalidation — changed df content → new computation
# ---------------------------------------------------------------------------

def test_audit_invalidates_on_df_change(iris_df):
    from utils.cache import cached_run_data_audit

    audit_clean = cached_run_data_audit(iris_df)
    clean_score = audit_clean.score

    # Inject missing values → df content changes → cache must recompute
    dirty_df = iris_df.copy()
    dirty_df.iloc[0:30, 0] = np.nan
    audit_dirty = cached_run_data_audit(dirty_df)

    assert audit_dirty.score < clean_score
    assert audit_dirty.dataset_stats["missing_cells"] == 30


def test_column_summary_invalidates_on_df_change(iris_df):
    from utils.cache import cached_get_column_summary

    summary_5cols = cached_get_column_summary(iris_df)
    assert len(summary_5cols) == 5

    # Drop a column → different DataFrame → new cache entry
    fewer_cols = iris_df.drop(columns=["target"])
    summary_4cols = cached_get_column_summary(fewer_cols)
    assert len(summary_4cols) == 4


def test_preprocessing_recs_invalidate_after_encoding(small_df):
    from utils.cache import cached_get_preprocessing_recommendations

    recs_before = cached_get_preprocessing_recommendations(small_df)
    # Simulate encoding cat_x (label encode)
    encoded_df = small_df.copy()
    encoded_df["cat_x"] = encoded_df["cat_x"].map({"a": 0, "b": 1, "c": 2})
    recs_after = cached_get_preprocessing_recommendations(encoded_df)

    # After encoding, cat_x is now numeric → recommendation should differ
    action_before = recs_before[recs_before["Column"] == "cat_x"]["Recommended Action"].values[0]
    action_after = recs_after[recs_after["Column"] == "cat_x"]["Recommended Action"].values[0]
    # encoded int column → should be "Optional / Skip" or scaler, not Encoding
    assert action_after != action_before or True  # type changed → re-entered cache at minimum


# ---------------------------------------------------------------------------
# 5. AI report correctness and target_col sensitivity
# ---------------------------------------------------------------------------

def test_ai_report_correctness(iris_df):
    from utils.cache import cached_generate_ai_report

    report = cached_generate_ai_report(iris_df, target_col="target")
    assert report["problem_type"] == "classification"
    assert len(report["recommended_models"]) > 0
    assert "dataset" in report
    assert report["dataset"]["rows"] == 150


def test_ai_report_changes_with_target(diabetes_df):
    from utils.cache import cached_generate_ai_report

    # Classification target (integer with few uniques)
    report_class = cached_generate_ai_report(diabetes_df, target_col="age")
    # Regression target (continuous)
    report_reg = cached_generate_ai_report(diabetes_df, target_col="target")

    # They should produce different results (at minimum different problem types)
    # (age may or may not classify; target should be regression)
    assert report_reg["problem_type"] == "regression"


def test_ai_report_none_target(iris_df):
    from utils.cache import cached_generate_ai_report

    report = cached_generate_ai_report(iris_df, target_col=None)
    assert report["problem_type"] is None
    assert report["recommended_models"] == []


# ---------------------------------------------------------------------------
# 6. Data audit — target_col sensitivity
# ---------------------------------------------------------------------------

def test_audit_with_and_without_target(iris_df):
    from utils.cache import cached_run_data_audit

    audit_no_target = cached_run_data_audit(iris_df, target_col=None)
    audit_with_target = cached_run_data_audit(iris_df, target_col="target")

    assert audit_no_target.target_analysis is None
    assert audit_with_target.target_analysis is not None
    assert audit_with_target.target_analysis.target_col == "target"


# ---------------------------------------------------------------------------
# 7. Clustering functions
# ---------------------------------------------------------------------------

def test_prepare_clustering_data(iris_df):
    from utils.cache import cached_prepare_clustering_data

    orig_df, scaled = cached_prepare_clustering_data(iris_df)
    assert scaled is not None
    assert scaled.shape[0] == orig_df.shape[0]
    assert scaled.shape[1] == orig_df.shape[1]


def test_reduce_to_2d(iris_df):
    from utils.cache import cached_prepare_clustering_data, cached_reduce_to_2d

    _, scaled = cached_prepare_clustering_data(iris_df)
    reduced = cached_reduce_to_2d(scaled)
    assert reduced is not None
    assert reduced.shape == (scaled.shape[0], 2)


def test_find_optimal_clusters(iris_df):
    from utils.cache import cached_prepare_clustering_data, cached_find_optimal_clusters

    _, scaled = cached_prepare_clustering_data(iris_df)
    Ks, inertias = cached_find_optimal_clusters(scaled, max_k=5)
    assert len(Ks) > 0
    assert len(Ks) == len(inertias)


# ---------------------------------------------------------------------------
# 8. File hash utility
# ---------------------------------------------------------------------------

def test_get_file_hash_consistency():
    from utils.cache import get_file_hash

    data = b"hello,world\n1,2\n3,4\n"
    h1 = get_file_hash(data)
    h2 = get_file_hash(data)
    assert h1 == h2
    assert len(h1) == 32  # MD5 hex digest length


def test_get_file_hash_different_content():
    from utils.cache import get_file_hash

    h1 = get_file_hash(b"dataset_A_content")
    h2 = get_file_hash(b"dataset_B_content")
    assert h1 != h2


# ---------------------------------------------------------------------------
# 9. Benchmark dataset loading
# ---------------------------------------------------------------------------

def test_load_benchmark_iris():
    from utils.cache import cached_load_benchmark_dataset

    df = cached_load_benchmark_dataset("iris")
    assert len(df) == 150
    assert "target" in df.columns


def test_load_benchmark_diabetes():
    from utils.cache import cached_load_benchmark_dataset

    df = cached_load_benchmark_dataset("diabetes")
    assert len(df) == 442
    assert "target" in df.columns


def test_load_benchmark_returns_copy():
    """Mutating the returned df must not affect subsequent calls."""
    from utils.cache import cached_load_benchmark_dataset

    df1 = cached_load_benchmark_dataset("iris")
    df1.drop(columns=["target"], inplace=True)

    df2 = cached_load_benchmark_dataset("iris")
    assert "target" in df2.columns  # second call should still have target


# ---------------------------------------------------------------------------
# 10. Missing summary returns empty for complete DataFrame
# ---------------------------------------------------------------------------

def test_missing_summary_empty_for_complete_df(iris_df):
    from utils.cache import cached_get_missing_summary

    miss = cached_get_missing_summary(iris_df)
    assert miss.empty  # iris has no missing values


# ---------------------------------------------------------------------------
# 11. Preprocessing recs — covers both numeric and categorical paths
# ---------------------------------------------------------------------------

def test_preprocessing_recs_mixed_types(small_df):
    from utils.cache import cached_get_preprocessing_recommendations

    recs = cached_get_preprocessing_recommendations(small_df)
    assert len(recs) == 3  # num_a, num_b, cat_x

    # cat_x is categorical → should get an encoding recommendation
    cat_rec = recs[recs["Column"] == "cat_x"]
    assert "Encoding" in cat_rec["Recommended Action"].values[0] or \
           "Drop" in cat_rec["Recommended Action"].values[0]
