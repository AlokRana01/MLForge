import pandas as pd
import numpy as np
import re


# ================= SAFE COPY =================
def safe_copy(df):
    return df.copy() if isinstance(df, pd.DataFrame) else None


# ================= CLEAN COLUMN NAMES =================
def clean_column_names(df):
    if df is None:
        return None

    df = df.copy()

    df.columns = [
        re.sub(r"[^\w]+", "_", col.strip().lower())
        for col in df.columns
    ]

    return df


# ================= CHECK EMPTY =================
def is_empty(df):
    return not isinstance(df, pd.DataFrame) or df.empty


# ================= SPLIT COLUMNS =================
def split_columns(df):
    if df is None:
        return [], []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return numeric_cols, categorical_cols


# ================= SAFE DIVISION =================
def safe_divide(a, b):
    try:
        return np.divide(a, b, where=(b != 0))
    except:
        return 0


# ================= FORMAT PERCENT =================
def format_percent(value):
    try:
        return f"{round(float(value) * 100, 2)}%"
    except:
        return "0%"


# ================= FORMAT NUMBER =================
def format_number(value):
    try:
        return round(float(value), 4)
    except:
        return 0


# ================= LIMIT DATAFRAME =================
def limit_df(df, rows=100):
    if not isinstance(df, pd.DataFrame):
        return None
    return df.head(rows).reset_index(drop=True)


# ================= SAFE DROP =================
def safe_drop_columns(df, cols):
    if df is None:
        return None

    df = df.copy()
    existing_cols = [c for c in cols if c in df.columns]

    return df.drop(columns=existing_cols)


# ================= ENCODE CATEGORICAL =================
def encode_categorical(df, max_unique=50):
    if df is None:
        return None

    df = df.copy()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        if df[col].nunique() > max_unique:
            df[col] = df[col].astype("category").cat.codes

    return pd.get_dummies(df, drop_first=True)


# ================= MEMORY USAGE =================
def get_memory_usage(df):
    try:
        return round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)
    except:
        return 0


# ================= UNIQUE SUMMARY =================
def get_unique_summary(df):
    if df is None:
        return {}

    return {col: int(df[col].nunique(dropna=False)) for col in df.columns}


# ================= SAFE TYPE CONVERSION =================
def try_numeric(df):
    if df is None:
        return None

    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


# ================= SAMPLE DATA =================
def sample_df(df, n=500):
    if not isinstance(df, pd.DataFrame):
        return None

    if len(df) <= n:
        return df.reset_index(drop=True)

    return df.sample(n, random_state=42).reset_index(drop=True)