import pandas as pd
import numpy as np


#  DATA OVERVIEW 
def analyze_dataset(df):
    if df is None or df.empty:
        return {
            "rows": 0,
            "columns": 0,
            "missing_cells": 0,
            "duplicate_rows": 0
        }

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }


#  PROBLEM TYPE 
def recommend_problem_type(df, target_col):
    if target_col not in df.columns:
        return None

    y = df[target_col]

    if y.dtype == "object":
        return "classification"

    if pd.api.types.is_numeric_dtype(y):
        unique_threshold = max(10, int(0.05 * len(y)))
        if y.nunique(dropna=True) < unique_threshold:
            return "classification"

    return "regression"


#  MISSING STRATEGY 
def recommend_missing_strategy(df):
    recommendations = {}

    for col in df.columns:
        col_data = df[col]

        missing_count = col_data.isna().sum()

        if missing_count == 0:
            continue

        # all null case
        if col_data.dropna().empty:
            recommendations[col] = "unknown"
            continue

        if pd.api.types.is_numeric_dtype(col_data):
            try:
                skew_val = col_data.skew()
                if abs(skew_val) > 1:
                    recommendations[col] = "median"
                else:
                    recommendations[col] = "mean"
            except:
                recommendations[col] = "mean"
        else:
            recommendations[col] = "unknown"

    return recommendations


#  FEATURE IMPORTANCE 
def recommend_features(df, target_col):
    if target_col not in df.columns:
        return []

    try:
        numeric_df = df.select_dtypes(include=[np.number])

        if target_col not in numeric_df.columns:
            return []

        corr = numeric_df.corr()[target_col]

        # remove NaN correlations
        corr = corr.dropna().abs().sort_values(ascending=False)

        # remove self
        corr = corr.drop(labels=[target_col], errors="ignore")

        return corr.head(5).index.tolist()

    except Exception as e:
        print(f"[ERROR] Feature recommendation failed: {e}")
        return []


#  MODEL RECOMMENDATION 
def recommend_models(problem_type):
    if problem_type == "classification":
        return [
            "LogisticRegression",
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "SVC",
            "KNN",
            "NaiveBayes"
        ]

    elif problem_type == "regression":
        return [
            "LinearRegression",
            "RandomForestRegressor",
            "DecisionTreeRegressor",
            "SVR"
        ]

    return []


#  CLUSTERING RECOMMENDATION 
def recommend_clustering(df):
    if df is None or df.empty:
        return {"recommended": None, "reason": "Empty dataset"}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        return {
            "recommended": None,
            "reason": "Not enough numeric features"
        }

    if len(df) < 50:
        return {
            "recommended": "Agglomerative",
            "reason": "Better for small datasets"
        }

    return {
        "recommended": "KMeans",
        "reason": "Efficient for larger datasets"
    }


#  EXPORT FORMAT 
def recommend_model_export(model):
    if model is None:
        return {"format": None, "reason": "No model provided"}

    try:
        name = model.__class__.__name__.lower()

        if hasattr(model, "named_steps"):
            # pipeline case
            name = str(model.named_steps).lower()

        if "forest" in name or "tree" in name:
            return {
                "format": "joblib",
                "reason": "Efficient for large sklearn models"
            }

        elif "linear" in name or "logistic" in name:
            return {
                "format": "pickle",
                "reason": "Lightweight and fast"
            }

        elif "svc" in name:
            return {
                "format": "joblib",
                "reason": "Better for SVM models"
            }

        return {
            "format": "pickle",
            "reason": "Default safe option"
        }

    except Exception as e:
        return {
            "format": "pickle",
            "reason": "Fallback option"
        }


#  FULL AI REPORT 
def generate_ai_report(df, target_col=None, model=None):
    report = {}

    report["dataset"] = analyze_dataset(df)

    if target_col and target_col in df.columns:
        problem_type = recommend_problem_type(df, target_col)

        report["problem_type"] = problem_type
        report["recommended_models"] = recommend_models(problem_type)
        report["important_features"] = recommend_features(df, target_col)

    else:
        report["problem_type"] = None
        report["recommended_models"] = []
        report["important_features"] = []

    report["missing_strategy"] = recommend_missing_strategy(df)
    report["clustering"] = recommend_clustering(df)

    if model:
        report["model_export"] = recommend_model_export(model)

    return report