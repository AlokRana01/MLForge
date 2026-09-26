import pandas as pd
import numpy as np


#  BASIC INFO 
def get_basic_info(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_cells": int(df.size),
        "duplicate_rows": int(df.duplicated().sum())
    }


#  COLUMN SUMMARY 
def get_column_summary(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    summary = []

    for col in df.columns:
        col_data = df[col]

        missing = col_data.isna().sum()
        total = len(col_data)

        try:
            top_value = col_data.mode(dropna=True)
            top_value = top_value.iloc[0] if not top_value.empty else None
        except:
            top_value = None

        summary.append({
            "column": col,
            "dtype": str(col_data.dtype),
            "non_null": int(col_data.count()),
            "missing": int(missing),
            "missing_%": round((missing / total) * 100, 2) if total > 0 else 0,
            "unique": int(col_data.nunique(dropna=False)),  # FIXED
            "top_value": top_value
        })

    return pd.DataFrame(summary)


#  NUMERIC STATS 
def get_numeric_stats(df):
    num_df = df.select_dtypes(include=[np.number])

    if num_df.empty:
        return pd.DataFrame()

    stats = num_df.describe().T

    # Extra metrics
    stats["median"] = num_df.median()
    stats["skew"] = num_df.skew()
    stats["kurtosis"] = num_df.kurt()

    return stats.reset_index().rename(columns={"index": "column"})


#  CATEGORICAL STATS 
def get_categorical_stats(df):
    cat_df = df.select_dtypes(include=["object", "category", "bool"])

    if cat_df.empty:
        return pd.DataFrame()

    data = []

    for col in cat_df.columns:
        col_data = cat_df[col]

        try:
            top = col_data.mode(dropna=True)
            top = top.iloc[0] if not top.empty else None
        except:
            top = None

        try:
            freq = col_data.value_counts(dropna=True)
            freq = int(freq.iloc[0]) if not freq.empty else None
        except:
            freq = None

        data.append({
            "column": col,
            "unique": int(col_data.nunique(dropna=False)),
            "top": top,
            "freq": freq
        })

    return pd.DataFrame(data)


#  MISSING MATRIX 
def get_missing_matrix(df, sample_size=500):
    # Large dataset optimization
    if len(df) > sample_size:
        return df.sample(sample_size).isna()
    return df.isna()


#  FULL REPORT 
def full_profiling_report(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    return {
        "basic_info": get_basic_info(df),
        "column_summary": get_column_summary(df),
        "numeric_stats": get_numeric_stats(df),
        "categorical_stats": get_categorical_stats(df),
        "missing_matrix": get_missing_matrix(df)
    }