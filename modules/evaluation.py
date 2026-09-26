"""
MLForge - Robust Model Evaluation Module
===============================================
Provides comprehensive evaluation metrics for classification and regression models:
- Cross-validation performance (CV Mean, CV Std)
- Holdout test set metrics
- Fault-tolerant handling of undefined metrics (ROC-AUC, PR-AUC, MAPE with zero true values)
- Configurable optimization metric sorting with higher-is-better / lower-is-better awareness
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
)
from sklearn.preprocessing import label_binarize


# =====================================================================
# METRIC SPECIFICATIONS & METADATA
# =====================================================================

METRIC_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Classification
    "F1 Score": {
        "problem_type": "classification",
        "higher_is_better": True,
        "default": True,
        "description": "Harmonic mean of precision and recall (weighted)"
    },
    "Accuracy": {
        "problem_type": "classification",
        "higher_is_better": True,
        "default": False,
        "description": "Overall percentage of correct predictions"
    },
    "Balanced Accuracy": {
        "problem_type": "classification",
        "higher_is_better": True,
        "default": False,
        "description": "Average recall per class (best for imbalanced data)"
    },
    "Precision": {
        "problem_type": "classification",
        "higher_is_better": True,
        "default": False,
        "description": "Weighted positive predictive value"
    },
    "Recall": {
        "problem_type": "classification",
        "higher_is_better": True,
        "default": False,
        "description": "Weighted sensitivity / true positive rate"
    },
    "ROC AUC": {
        "problem_type": "classification",
        "higher_is_better": True,
        "default": False,
        "description": "Area under the Receiver Operating Characteristic curve"
    },
    "PR AUC": {
        "problem_type": "classification",
        "higher_is_better": True,
        "default": False,
        "description": "Area under the Precision-Recall curve (Average Precision)"
    },

    # Regression
    "R2 Score": {
        "problem_type": "regression",
        "higher_is_better": True,
        "default": True,
        "description": "Coefficient of determination (variance explained)"
    },
    "RMSE": {
        "problem_type": "regression",
        "higher_is_better": False,
        "default": False,
        "description": "Root Mean Squared Error (penalizes large errors)"
    },
    "MAE": {
        "problem_type": "regression",
        "higher_is_better": False,
        "default": False,
        "description": "Mean Absolute Error (interpretable scale)"
    },
    "MAPE": {
        "problem_type": "regression",
        "higher_is_better": False,
        "default": False,
        "description": "Mean Absolute Percentage Error (%)"
    }
}


def get_available_metrics(problem_type: str) -> List[str]:
    """Return valid metric names for the given task type."""
    return [
        name for name, cfg in METRIC_CONFIGS.items()
        if cfg["problem_type"] == problem_type
    ]


def is_higher_better(metric_name: Optional[str]) -> bool:
    """Check if higher values are better for a given metric."""
    if not metric_name or not isinstance(metric_name, str):
        return True
    cfg = METRIC_CONFIGS.get(metric_name)
    if cfg:
        return cfg["higher_is_better"]
    # Default heuristics
    if any(k in metric_name.upper() for k in ("RMSE", "MAE", "MAPE", "LOSS", "ERROR")):
        return False
    return True


def get_sklearn_scoring(metric_name: str, problem_type: str = "classification") -> str:
    """
    Map user metric names to standard scikit-learn scoring parameters
    for use in cross_val_score, GridSearchCV, and RandomizedSearchCV.
    """
    m_clean = metric_name.strip()
    mapping = {
        "F1 Score": "f1_weighted",
        "Accuracy": "accuracy",
        "Balanced Accuracy": "balanced_accuracy",
        "Precision": "precision_weighted",
        "Recall": "recall_weighted",
        "ROC AUC": "roc_auc_ovr" if problem_type == "classification" else "roc_auc",
        "PR AUC": "average_precision",
        "R2 Score": "r2",
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "MAPE": "neg_mean_absolute_percentage_error",
    }
    return mapping.get(m_clean, "f1_weighted" if problem_type == "classification" else "r2")


# =====================================================================
# INPUT VALIDATION
# =====================================================================

def _validate_inputs(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical length")


# =====================================================================
# CLASSIFICATION EVALUATION
# =====================================================================

def evaluate_classification(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    y_proba: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    classes: Optional[Union[np.ndarray, list]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification performance metrics with fault-tolerant
    handling for edge cases (undefined division, single class present in test, etc.).
    """
    _validate_inputs(y_true, y_pred)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    result: Dict[str, Any] = {
        "Accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "Precision": round(float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)), 4),
        "F1 Score": round(float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)), 4),
        "Balanced Accuracy": round(float(balanced_accuracy_score(y_true_arr, y_pred_arr)), 4),
        "ROC AUC": None,
        "PR AUC": None
    }

    # Safe calculation for ROC AUC & PR AUC
    if y_proba is not None:
        try:
            proba_arr = np.asarray(y_proba)
            unique_labels = np.unique(y_true_arr)
            n_classes = len(unique_labels)

            if n_classes < 2:
                # Undefined if only 1 class in test set
                result["ROC AUC"] = None
                result["PR AUC"] = None
            elif n_classes == 2:
                # Binary classification
                # Use second column if 2D
                if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                    pos_proba = proba_arr[:, 1]
                else:
                    pos_proba = proba_arr.ravel()

                # Convert binary labels to 0/1 if not already
                if unique_labels.tolist() == [0, 1]:
                    bin_y = y_true_arr
                else:
                    bin_y = (y_true_arr == unique_labels[1]).astype(int)

                result["ROC AUC"] = round(float(roc_auc_score(bin_y, pos_proba)), 4)
                result["PR AUC"] = round(float(average_precision_score(bin_y, pos_proba)), 4)
            else:
                # Multi-class
                if proba_arr.ndim == 2 and proba_arr.shape[1] >= n_classes:
                    all_classes = classes if classes is not None else np.unique(np.concatenate([y_true_arr, y_pred_arr]))
                    # Binarize labels
                    bin_y = label_binarize(y_true_arr, classes=all_classes)
                    if bin_y.shape[1] == proba_arr.shape[1]:
                        result["ROC AUC"] = round(float(roc_auc_score(bin_y, proba_arr, multi_class="ovr", average="weighted")), 4)
                        result["PR AUC"] = round(float(average_precision_score(bin_y, proba_arr, average="weighted")), 4)
        except Exception:
            # Silently handle undefined metric
            result["ROC AUC"] = None
            result["PR AUC"] = None

    return result


# =====================================================================
# REGRESSION EVALUATION
# =====================================================================

def evaluate_regression(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list]
) -> Dict[str, Any]:
    """
    Compute comprehensive regression performance metrics with fault-tolerant
    handling for edge cases (zero values in true labels, constant target, etc.).
    """
    _validate_inputs(y_true, y_pred)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # R2 Score
    try:
        r2 = round(float(r2_score(y_true_arr, y_pred_arr)), 4)
    except Exception:
        r2 = 0.0

    # MAE
    try:
        mae = round(float(mean_absolute_error(y_true_arr, y_pred_arr)), 4)
    except Exception:
        mae = 0.0

    # RMSE
    try:
        rmse = round(float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))), 4)
    except Exception:
        rmse = 0.0

    # MAPE (defensive against zero division)
    try:
        mask = np.abs(y_true_arr) > 1e-8
        if np.any(mask):
            mape_val = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100.0
            mape = round(float(mape_val), 4)
        else:
            mape = 0.0
    except Exception:
        mape = 0.0

    return {
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


# =====================================================================
# UNIFIED EVALUATION DISPATCHER
# =====================================================================

def evaluate_model(
    y_true: Any,
    y_pred: Any,
    problem_type: str,
    y_proba: Optional[Any] = None,
    classes: Optional[Any] = None
) -> Dict[str, Any]:
    """Evaluate predictions according to problem type."""
    if problem_type == "classification":
        return evaluate_classification(y_true, y_pred, y_proba, classes=classes)
    elif problem_type == "regression":
        return evaluate_regression(y_true, y_pred)
    return {}


# =====================================================================
# LEADERBOARD / RESULTS TABLE
# =====================================================================

def build_results_table(
    results_list: List[Dict[str, Any]],
    sort_metric: Optional[str] = None,
    problem_type: str = "classification"
) -> pd.DataFrame:
    """
    Build a structured, sorted leaderboard table from model results.
    Respects higher_is_better vs lower_is_better for user-configured optimization metric.
    """
    if not results_list:
        return pd.DataFrame()

    df = pd.DataFrame(results_list)

    # Determine sorting metric
    if sort_metric is None or sort_metric not in df.columns:
        if problem_type == "classification":
            sort_metric = "F1 Score" if "F1 Score" in df.columns else "Accuracy"
        else:
            sort_metric = "R2 Score" if "R2 Score" in df.columns else "RMSE"

    if sort_metric in df.columns:
        higher_better = is_higher_better(sort_metric)
        df = df.sort_values(by=sort_metric, ascending=not higher_better).reset_index(drop=True)

    # Reorder key summary columns up front for clean presentation
    priority_cols = [
        "Model", "Stage", "Optimization Status", "CV Mean", "CV Std",
        "Test Score", "Train Time (s)", "Best Hyperparameters"
    ]
    front_cols = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    return df