import pandas as pd


#  DETECT DUPLICATES 
def get_duplicate_summary(df, subset=None):
    if df is None or df.empty:
        return {
            "total_duplicates": 0,
            "duplicate_%": 0,
        }

    # Validate subset
    if subset:
        subset = [col for col in subset if col in df.columns]
        if not subset:
            subset = None

    dup_mask = df.duplicated(subset=subset, keep="first")

    total_dup = int(dup_mask.sum())
    total_rows = len(df)

    return {
        "total_duplicates": total_dup,
        "duplicate_%": round((total_dup / total_rows) * 100, 2) if total_rows > 0 else 0,
    }


#  PREVIEW DUPLICATES 
def get_duplicate_rows(df, subset=None, limit=100):
    if df is None or df.empty:
        return pd.DataFrame()

    # Validate subset
    if subset:
        subset = [col for col in subset if col in df.columns]
        if not subset:
            subset = None

    dup_df = df[df.duplicated(subset=subset, keep=False)]

    return dup_df.head(limit)


#  REMOVE DUPLICATES 
def remove_duplicates(df, subset=None, keep="first"):
    """
    keep options:
    - "first"
    - "last"
    - False (remove all duplicates)
    """

    if df is None or df.empty:
        return df, {"before": 0, "after": 0, "removed": 0}

    df = df.copy()

    # Validate subset
    if subset:
        subset = [col for col in subset if col in df.columns]
        if not subset:
            subset = None

    before = len(df)

    df_clean = df.drop_duplicates(subset=subset, keep=keep)

    after = len(df_clean)

    removed = before - after

    return df_clean, {
        "before": before,
        "after": after,
        "removed": removed
    }


#  SMART SUGGESTION 
def suggest_duplicate_strategy(df):
    if df is None or df.empty:
        return {
            "subset": None,
            "keep": "first"
        }

    total_rows = len(df)

    if total_rows == 0:
        return {
            "subset": None,
            "keep": "first"
        }

    # Include NaN in uniqueness calculation
    unique_ratio = df.nunique(dropna=False) / total_rows

    # Exclude high-unique columns (like IDs)
    subset_cols = unique_ratio[unique_ratio < 0.9].index.tolist()

    if not subset_cols:
        subset_cols = None

    return {
        "subset": subset_cols,
        "keep": "first"
    }