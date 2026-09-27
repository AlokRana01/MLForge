"""
MLForge — Centralized Caching Layer
====================================
Production-grade caching for all expensive, deterministic operations.

Cache Architecture:
  - st.cache_data  → pure, deterministic functions (data in, data out)
  - st.cache_resource → not used (no shared singleton resources in MLForge)
  - st.session_state → user-specific mutable workflow state (managed in app.py)

Safety rules enforced here:
  1. Cached functions never mutate their inputs.
  2. File identity is based on content hash, never filename alone.
  3. DataFrames passed to cached functions are hashed by Streamlit on content.
  4. max_entries guards against unbounded memory growth.
  5. No TTL set — results are deterministic and never become stale on their own.
"""

import hashlib
import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# FILE LOADING
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=8, show_spinner=False)
def cached_parse_uploaded_file(file_bytes: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Parse a user-uploaded file into a DataFrame.

    Cache key: file_bytes content (Streamlit hashes this automatically) + filename
    (filename is included so that two files with different extensions but same
    content still dispatch to the correct parser branch).

    Returns a copy of the parsed DataFrame so the caller can freely mutate it
    without affecting the cached result.
    """
    from modules.file_loader import load_file
    import io

    # Wrap bytes back into a file-like object that mimics UploadedFile
    class _BytesFile:
        """Minimal file-like wrapper that satisfies load_file()'s interface."""
        def __init__(self, data: bytes, fname: str):
            self._buf = io.BytesIO(data)
            self.name = fname

        def seek(self, pos: int) -> None:
            self._buf.seek(pos)

        def read(self, *args) -> bytes:
            return self._buf.read(*args)

        def __iter__(self):
            return iter(self._buf)

    proxy = _BytesFile(file_bytes, filename)
    df, msg = load_file(proxy)

    if df is not None:
        return df.copy(), msg
    return None, msg


@st.cache_data(max_entries=8, show_spinner=False)
def cached_load_benchmark_dataset(bench_id: str) -> pd.DataFrame:
    """
    Load a standard sklearn benchmark dataset and return a copy.
    These never change — safe to cache indefinitely.
    """
    if bench_id == "iris":
        from sklearn.datasets import load_iris
        ds = load_iris(as_frame=True)
    elif bench_id == "wine":
        from sklearn.datasets import load_wine
        ds = load_wine(as_frame=True)
    elif bench_id == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        ds = load_breast_cancer(as_frame=True)
    elif bench_id == "diabetes":
        from sklearn.datasets import load_diabetes
        ds = load_diabetes(as_frame=True)
    else:
        raise ValueError(f"Unknown benchmark id: {bench_id}")
    return ds.frame.copy()


def get_file_hash(file_bytes: bytes) -> str:
    """
    Compute a stable MD5 hex digest for an uploaded file's raw bytes.
    Use this as a supplementary cache key when you need a human-readable
    string representation of file identity (e.g., logging).

    NOTE: Streamlit's @st.cache_data already hashes bytes arguments
    automatically — this utility is for explicit use in session_state comparisons.
    """
    return hashlib.md5(file_bytes).hexdigest()


# ---------------------------------------------------------------------------
# DATA AUDIT
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=32, show_spinner=False)
def cached_run_data_audit(df: pd.DataFrame, target_col: Optional[str] = None):
    """
    Run the comprehensive ML Readiness Audit on a DataFrame.

    This is one of the most expensive per-rerun operations in MLForge:
    it performs per-column outlier detection (IQR), cardinality analysis,
    type inference, leakage detection, and computes the readiness score.

    Cache invalidation:
      - Automatic: df content changes (cleaning, encoding, etc.) → new hash
      - Automatic: target_col changes → new cache entry

    Returns the AuditResult dataclass directly (not mutated by callers).
    """
    from modules.data_audit import run_data_audit
    return run_data_audit(df, target_col=target_col)


# ---------------------------------------------------------------------------
# PROFILING
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=32, show_spinner=False)
def cached_get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic dataset shape information: rows, columns, total_cells, duplicate_rows.
    Called on Steps 2, 5, and 9 (report).
    """
    from modules.profiling import get_basic_info
    return get_basic_info(df)


@st.cache_data(max_entries=32, show_spinner=False)
def cached_get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column summary: dtype, non_null, missing count/%, unique, top value.
    Called on Steps 2, 5, and 9 (report).
    Returns a copy — callers can safely display or modify.
    """
    from modules.profiling import get_column_summary
    return get_column_summary(df)


@st.cache_data(max_entries=32, show_spinner=False)
def cached_get_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute extended numeric statistics: describe() + median, skew, kurtosis.
    Called on Steps 2 and 9.
    """
    from modules.profiling import get_numeric_stats
    return get_numeric_stats(df)


@st.cache_data(max_entries=32, show_spinner=False)
def cached_get_categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute categorical column statistics: unique count, top value, frequency.
    Called on Step 2.
    """
    from modules.profiling import get_categorical_stats
    return get_categorical_stats(df)


@st.cache_data(max_entries=32, show_spinner=False)
def cached_compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix for all numeric columns.
    This is O(cols² × rows) — expensive on wide datasets.
    Called on Steps 2 (heatmap) and 9 (Excel/PDF/Word reports).

    Only numeric columns are used; the caller should filter before passing in
    if they want a specific subset.
    """
    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) < 2:
        return pd.DataFrame()
    return num_df.corr()


# ---------------------------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=32, show_spinner=False)
def cached_get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column missing value statistics: count, percentage, dtype.
    Called every rerun on Step 3.
    """
    from modules.missing_handler import get_missing_summary
    return get_missing_summary(df)


@st.cache_data(max_entries=32, show_spinner=False)
def cached_suggest_missing_strategy(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer imputation strategy per column (mean/median/unknown).
    Based on skewness — deterministic for a given DataFrame.
    Called every rerun on Step 3.
    """
    from modules.missing_handler import suggest_missing_strategy
    return suggest_missing_strategy(df)


# ---------------------------------------------------------------------------
# PREPROCESSING RECOMMENDATIONS
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=32, show_spinner=False)
def cached_get_preprocessing_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the AI preprocessing recommendation table for Step 5.

    This replaces the per-column loop that runs on every rerun of Step 5.
    The loop computes: range, std, skew per numeric column and
    cardinality per categorical column.

    Cache invalidation:
      - Automatic when df changes (encoding / scaling / column drop alters content).
    """
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        is_num = pd.api.types.is_numeric_dtype(df[col])
        is_cat = dtype in ("object", "category") or not is_num

        if is_num:
            col_data = df[col].dropna()
            try:
                rng = float(col_data.max() - col_data.min())
                skew = float(col_data.skew())
                std = float(col_data.std())
            except Exception:
                rng = skew = std = 0

            if rng > 100 or std > 10:
                if abs(skew) > 1:
                    action = "RobustScaler"
                    reason = f"High skew ({skew:.2f}) + large range ({rng:.0f}) — robust to outliers"
                else:
                    action = "StandardScaler"
                    reason = f"Large range ({rng:.0f}), std={std:.2f} — normalize to zero mean"
            elif rng > 1:
                action = "MinMaxScaler"
                reason = f"Moderate range ({rng:.2f}) — scale to 0-1"
            else:
                action = "Optional / Skip"
                reason = "Already small range — scaling may not be needed"
        else:
            if n_unique > 50:
                action = "Drop or Hash"
                reason = f"Very high cardinality ({n_unique} unique) — likely an ID or free-text, drop it"
            elif n_unique > 10:
                action = "Label Encoding"
                reason = f"High cardinality ({n_unique} unique) — label encoding preferred"
            elif n_unique > 2:
                action = "One-Hot Encoding"
                reason = f"Low cardinality ({n_unique} unique) — safe for one-hot"
            else:
                action = "Label Encoding"
                reason = f"Binary / bool ({n_unique} unique) — simple label encoding"

        rows.append({
            "Column": col,
            "Type": dtype,
            "Unique Values": n_unique,
            "Recommended Action": action,
            "Reason": reason,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AI RECOMMENDER
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=32, show_spinner=False)
def cached_generate_ai_report(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate the full AI report: dataset stats, problem type, recommended models,
    important features (correlation-based), missing strategy, clustering recommendation.

    This function calls df.corr() internally (O(cols²×rows)) and is called on
    every rerun of Step 7 — one of the heaviest repeated costs.

    Cache invalidation:
      - Automatic when df content changes
      - Automatic when target_col changes
    """
    from modules.ai_recommender import generate_ai_report
    return generate_ai_report(df, target_col=target_col)


# ---------------------------------------------------------------------------
# CLUSTERING
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=8, show_spinner=False)
def cached_prepare_clustering_data(df: pd.DataFrame):
    """
    Scale numeric data for clustering (StandardScaler.fit_transform).
    Returns (original_numeric_df, scaled_array) — callers must copy before mutation.

    Cache invalidation:
      - Automatic when df content changes (new upload, cleaning, encoding).
    """
    from modules.clustering import prepare_clustering_data
    orig_df, scaled = prepare_clustering_data(df)
    # Return scaled as bytes for stable caching (numpy arrays are mutable)
    # Streamlit hashes numpy arrays by content — this is safe as-is.
    return orig_df, scaled


@st.cache_data(max_entries=8, show_spinner=False)
def cached_reduce_to_2d(scaled: np.ndarray) -> Optional[np.ndarray]:
    """
    Reduce scaled clustering data to 2 PCA components for visualization.
    PCA is O(n_samples × n_features²) — worth caching even for moderate datasets.

    Cache invalidation:
      - Automatic when scaled array content changes.
    """
    from modules.clustering import reduce_to_2d
    return reduce_to_2d(scaled)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_find_optimal_clusters(scaled: np.ndarray, max_k: int = 10) -> Tuple[list, list]:
    """
    Compute KMeans inertia across k=2..max_k for the elbow curve.
    This is invoked inside a Streamlit expander on Step 8 — previously
    re-ran on every expand/collapse interaction.

    Cache invalidation:
      - Automatic when scaled content changes.
    """
    from modules.clustering import find_optimal_clusters
    return find_optimal_clusters(scaled, max_k=max_k)


# ---------------------------------------------------------------------------
# LOGO / STATIC ASSETS
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=1, show_spinner=False)
def cached_load_logo_svg(logo_path: str) -> str:
    """
    Read the MLForge SVG logo from disk once and cache it in memory.
    The logo is read on every rerun for the sidebar — this eliminates
    redundant file I/O.

    Cache invalidation:
      - Automatic when logo_path changes (path is the key).
      - In practice this never changes during a session.
    """
    try:
        with open(logo_path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""
