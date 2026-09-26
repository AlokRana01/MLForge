import pandas as pd
import numpy as np


#  DETECT MISSING 
def get_missing_summary(df):
    if df is None or df.empty:
        return pd.DataFrame()

    summary = []

    total_rows = len(df)

    for col in df.columns:
        col_data = df[col]
        missing = col_data.isna().sum()

        if missing > 0:
            summary.append({
                "column": col,
                "missing_count": int(missing),
                "missing_%": round((missing / total_rows) * 100, 2) if total_rows > 0 else 0,
                "dtype": str(col_data.dtype)
            })

    return pd.DataFrame(summary)


#  APPLY STRATEGY 
def fill_missing_values(df, strategy_map, custom_values=None):
    """
    strategy_map example:
    {
        "age": "mean",
        "salary": "median",
        "city": "unknown",
        "experience": "ffill",
        "rating": "constant"
    }

    custom_values example:
    {
        "rating": 5
    }
    """

    df = df.copy()
    custom_values = custom_values or {}

    for col, strategy in strategy_map.items():

        if col not in df.columns:
            continue

        col_data = df[col]

        # Skip if no missing
        if col_data.isna().sum() == 0:
            continue

        try:
            # ===== NUMERIC =====
            if pd.api.types.is_numeric_dtype(col_data):

                if strategy == "mean":
                    val = col_data.mean()
                    if not np.isnan(val):
                        df[col] = col_data.fillna(val)

                elif strategy == "median":
                    val = col_data.median()
                    if not np.isnan(val):
                        df[col] = col_data.fillna(val)

                elif strategy == "mode":
                    mode_val = col_data.mode(dropna=True)
                    if not mode_val.empty:
                        df[col] = col_data.fillna(mode_val.iloc[0])

                elif strategy == "ffill":
                    df[col] = col_data.ffill()

                elif strategy == "bfill":
                    df[col] = col_data.bfill()

                elif strategy == "zero":
                    df[col] = col_data.fillna(0)

                elif strategy == "constant":
                    if col in custom_values:
                        df[col] = col_data.fillna(custom_values[col])

            # ===== CATEGORICAL =====
            else:

                if strategy == "unknown":
                    df[col] = col_data.fillna("Unknown")

                elif strategy == "mode":
                    mode_val = col_data.mode(dropna=True)
                    if not mode_val.empty:
                        df[col] = col_data.fillna(mode_val.iloc[0])

                elif strategy == "ffill":
                    df[col] = col_data.ffill()

                elif strategy == "bfill":
                    df[col] = col_data.bfill()

                elif strategy == "empty":
                    df[col] = col_data.fillna("")

                elif strategy == "constant":
                    if col in custom_values:
                        df[col] = col_data.fillna(custom_values[col])

        except Exception as e:
            print(f"[WARNING] Column '{col}' skipped due to error: {e}")

    return df


#  AUTO STRATEGY 
def suggest_missing_strategy(df):
    suggestions = {}

    for col in df.columns:
        col_data = df[col]

        if col_data.isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(col_data):

            # Handle all-null case
            if col_data.dropna().empty:
                suggestions[col] = "zero"
                continue

            skewness = col_data.skew()

            if abs(skewness) > 1:
                suggestions[col] = "median"
            else:
                suggestions[col] = "mean"

        else:
            suggestions[col] = "unknown"

    return suggestions