"""
MLForge - Explainable AI & Model Interpretability Engine
==============================================================
Provides statistically grounded explainability for machine learning pipelines:
- Inspects model architecture across Tree, Linear, Kernel/Distance, and Probabilistic families.
- Computes global feature importance using native mechanisms (tree feature_importances_,
  linear absolute coefficients) and scikit-learn permutation importance fallback.
- Integrates SHAP with TreeExplainer, LinearExplainer, and generic fallback explainers.
- Implements strict performance safeguards (sample size limits, background caching).
- Explains individual predictions with local feature attributions and ethical AI
  non-causality disclaimers.
- Renders publication-grade interactive Plotly visualizations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Lazy SHAP import to eliminate 1.5s - 3.5s startup penalty
_shap_instance = None
_shap_checked = False


def _ensure_shap() -> Any:
    """Lazily load the SHAP library when explainability routines are executed."""
    global _shap_instance, _shap_checked
    if not _shap_checked:
        _shap_checked = True
        try:
            import shap as _s
            _shap_instance = _s
        except Exception:
            _shap_instance = None
    return _shap_instance


def _has_shap() -> bool:
    """Return True if SHAP is available in the environment."""
    return _ensure_shap() is not None


def __getattr__(name: str) -> Any:
    """Support module-level access to _HAS_SHAP and shap without eager startup imports."""
    if name == "_HAS_SHAP":
        return _has_shap()
    if name == "shap":
        return _ensure_shap()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# =====================================================================
# MODEL FAMILY INSPECTION
# =====================================================================

def get_underlying_model(pipeline_or_model: Any) -> Any:
    """
    Extract the underlying estimator from an sklearn Pipeline or return the model.
    """
    if isinstance(pipeline_or_model, Pipeline):
        if "model" in pipeline_or_model.named_steps:
            return pipeline_or_model.named_steps["model"]
        return pipeline_or_model.steps[-1][1]
    return pipeline_or_model


def get_preprocessor(pipeline_or_model: Any) -> Optional[Any]:
    """
    Extract the ColumnTransformer / preprocessor from a Pipeline if present.
    """
    if isinstance(pipeline_or_model, Pipeline):
        if "preprocessor" in pipeline_or_model.named_steps:
            return pipeline_or_model.named_steps["preprocessor"]
        if len(pipeline_or_model.steps) > 1:
            return pipeline_or_model.steps[0][1]
    return None


def inspect_model_family(model_obj: Any) -> Dict[str, Any]:
    """
    Inspect the model architecture and determine supported explanation mechanisms.
    """
    raw_model = get_underlying_model(model_obj)
    cls_name = raw_model.__class__.__name__

    info = {
        "model_name": cls_name,
        "raw_model": raw_model,
        "family": "Unknown",
        "has_native_importance": False,
        "native_importance_type": None,
        "shap_explainer_type": None,
        "supports_shap": False,
        "supports_probabilities": hasattr(raw_model, "predict_proba"),
    }

    # Tree-based architectures
    tree_classes = (
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "HistGradientBoostingClassifier", "HistGradientBoostingRegressor"
    )

    # Linear architectures
    linear_classes = (
        "LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet"
    )

    # Kernel, distance, and probabilistic architectures
    kernel_distance_classes = ("SVC", "SVR", "KNeighborsClassifier", "GaussianNB")
    has_shap = _has_shap()

    if cls_name in tree_classes:
        info["family"] = "Tree"
        if hasattr(raw_model, "feature_importances_"):
            info["has_native_importance"] = True
            info["native_importance_type"] = "feature_importances_"
        info["shap_explainer_type"] = "TreeExplainer"
        info["supports_shap"] = has_shap

    elif cls_name in linear_classes:
        info["family"] = "Linear"
        if hasattr(raw_model, "coef_"):
            info["has_native_importance"] = True
            info["native_importance_type"] = "coef_"
        info["shap_explainer_type"] = "LinearExplainer"
        info["supports_shap"] = has_shap

    elif cls_name in kernel_distance_classes:
        info["family"] = "Kernel/Distance/Probabilistic"
        info["has_native_importance"] = False
        info["native_importance_type"] = None
        info["shap_explainer_type"] = "KernelExplainer"
        info["supports_shap"] = has_shap

    else:
        # Generic heuristic fallback
        if hasattr(raw_model, "feature_importances_"):
            info["family"] = "Tree"
            info["has_native_importance"] = True
            info["native_importance_type"] = "feature_importances_"
            info["shap_explainer_type"] = "TreeExplainer"
            info["supports_shap"] = has_shap
        elif hasattr(raw_model, "coef_"):
            info["family"] = "Linear"
            info["has_native_importance"] = True
            info["native_importance_type"] = "coef_"
            info["shap_explainer_type"] = "LinearExplainer"
            info["supports_shap"] = has_shap
        else:
            info["family"] = "Other"
            info["has_native_importance"] = False
            info["shap_explainer_type"] = "KernelExplainer"
            info["supports_shap"] = has_shap

    return info


# =====================================================================
# FEATURE NAMES EXTRACTION & MAPPING
# =====================================================================

def clean_transformed_feature_name(name: str) -> str:
    """
    Remove ColumnTransformer prefixes like 'num__' or 'cat__' for cleaner display.
    """
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def get_feature_names_from_pipeline(
    pipeline_or_model: Any,
    raw_feature_names: Optional[List[str]] = None
) -> List[str]:
    """
    Retrieve feature names matching the transformed input dimensionality of the estimator.
    """
    preprocessor = get_preprocessor(pipeline_or_model)
    if preprocessor is not None:
        try:
            if hasattr(preprocessor, "get_feature_names_out"):
                raw_names = preprocessor.get_feature_names_out()
                return [clean_transformed_feature_name(n) for n in raw_names]
        except Exception:
            pass

    if raw_feature_names is not None:
        return list(raw_feature_names)

    # Fallback to generic feature indices
    raw_model = get_underlying_model(pipeline_or_model)
    n_features = getattr(raw_model, "n_features_in_", None)
    if n_features is not None:
        return [f"Feature_{i}" for i in range(n_features)]

    return []


# =====================================================================
# GLOBAL FEATURE IMPORTANCE (NATIVE + PERMUTATION FALLBACK)
# =====================================================================

def compute_global_feature_importance(
    pipeline_or_model: Any,
    X_sample: Optional[pd.DataFrame] = None,
    y_sample: Optional[Union[pd.Series, np.ndarray]] = None,
    raw_feature_names: Optional[List[str]] = None,
    problem_type: str = "classification",
    n_repeats: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute global feature importance with transparent mechanism disclosure:
    1. Native tree feature_importances_
    2. Native linear absolute coefficients (|coef_|)
    3. Permutation importance fallback across raw input features when native is unavailable
    4. Explicitly avoids inventing arbitrary importance for unsupported models without data.
    """
    raw_model = get_underlying_model(pipeline_or_model)
    preprocessor = get_preprocessor(pipeline_or_model)
    model_info = inspect_model_family(raw_model)

    feat_names = get_feature_names_from_pipeline(pipeline_or_model, raw_feature_names)

    # --- Case 1: Native Tree Importance ---
    if hasattr(raw_model, "feature_importances_"):
        raw_importances = np.array(raw_model.feature_importances_, dtype=float)
        if len(feat_names) == len(raw_importances):
            names = feat_names
        else:
            names = [f"Feature_{i}" for i in range(len(raw_importances))]

        df_imp = pd.DataFrame({
            "Feature": names,
            "Importance": raw_importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        # Normalize to 0-100%
        total = df_imp["Importance"].sum()
        df_imp["Relative_Weight_%"] = (df_imp["Importance"] / total * 100.0) if total > 0 else 0.0

        return {
            "status": "success",
            "mechanism": "Native Tree Gini / Impurity (feature_importances_)",
            "model_family": model_info["family"],
            "model_type": model_info["model_name"],
            "importance_df": df_imp,
            "top_features": df_imp["Feature"].head(10).tolist(),
            "is_native": True
        }

    # --- Case 2: Native Linear Coefficients ---
    if hasattr(raw_model, "coef_"):
        raw_coefs = np.array(raw_model.coef_, dtype=float)
        # Multiclass or multioutput has shape (n_classes, n_features)
        if raw_coefs.ndim > 1:
            importance_vals = np.mean(np.abs(raw_coefs), axis=0)
        else:
            importance_vals = np.abs(raw_coefs)

        if len(feat_names) == len(importance_vals):
            names = feat_names
        else:
            names = [f"Feature_{i}" for i in range(len(importance_vals))]

        df_imp = pd.DataFrame({
            "Feature": names,
            "Importance": importance_vals
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        total = df_imp["Importance"].sum()
        df_imp["Relative_Weight_%"] = (df_imp["Importance"] / total * 100.0) if total > 0 else 0.0

        return {
            "status": "success",
            "mechanism": "Linear Model Absolute Coefficients (|coef_| magnitude)",
            "model_family": model_info["family"],
            "model_type": model_info["model_name"],
            "importance_df": df_imp,
            "top_features": df_imp["Feature"].head(10).tolist(),
            "is_native": True
        }

    # --- Case 3: Permutation Importance Fallback on Validation Data ---
    if X_sample is not None and y_sample is not None and len(X_sample) >= 5:
        try:
            # Run permutation importance on the entire pipeline with raw features
            perm_result = permutation_importance(
                pipeline_or_model,
                X_sample,
                y_sample,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=1
            )
            raw_cols = X_sample.columns.tolist() if isinstance(X_sample, pd.DataFrame) else [f"Feature_{i}" for i in range(X_sample.shape[1])]
            mean_importances = np.maximum(0.0, perm_result.importances_mean)

            df_imp = pd.DataFrame({
                "Feature": raw_cols,
                "Importance": mean_importances,
                "Importance_Std": perm_result.importances_std
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

            total = df_imp["Importance"].sum()
            df_imp["Relative_Weight_%"] = (df_imp["Importance"] / total * 100.0) if total > 0 else 0.0

            return {
                "status": "success",
                "mechanism": "Permutation Feature Importance (Validation Set Perturbation)",
                "model_family": model_info["family"],
                "model_type": model_info["model_name"],
                "importance_df": df_imp,
                "top_features": df_imp["Feature"].head(10).tolist(),
                "is_native": False
            }
        except Exception as e:
            return {
                "status": "error",
                "mechanism": "Permutation Importance (Failed)",
                "model_family": model_info["family"],
                "model_type": model_info["model_name"],
                "importance_df": pd.DataFrame(columns=["Feature", "Importance", "Relative_Weight_%"]),
                "top_features": [],
                "is_native": False,
                "message": f"Permutation importance failed: {e}"
            }

    # --- Case 4: Unsupported model without validation data ---
    return {
        "status": "unsupported",
        "mechanism": "None",
        "model_family": model_info["family"],
        "model_type": model_info["model_name"],
        "importance_df": pd.DataFrame(columns=["Feature", "Importance", "Relative_Weight_%"]),
        "top_features": [],
        "is_native": False,
        "message": (
            f"Native feature importance is not available for {model_info['model_name']} "
            f"({model_info['family']}). Provide validation samples to compute Permutation Importance."
        )
    }


# =====================================================================
# SHAP EXPLAINABILITY ENGINE
# =====================================================================

def prepare_data_for_shap(
    pipeline_or_model: Any,
    X_raw: pd.DataFrame,
    max_samples: int = 100
) -> Tuple[np.ndarray, List[str]]:
    """
    Subsample and transform raw tabular data through the pipeline's preprocessor.
    """
    if X_raw is None or len(X_raw) == 0:
        raise ValueError("Cannot prepare data for SHAP: input DataFrame is empty.")

    sample_df = X_raw.sample(n=min(len(X_raw), max_samples), random_state=42) if len(X_raw) > max_samples else X_raw.copy()
    preprocessor = get_preprocessor(pipeline_or_model)

    if preprocessor is not None:
        X_trans = preprocessor.transform(sample_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        feat_names = get_feature_names_from_pipeline(pipeline_or_model, sample_df.columns.tolist())
    else:
        X_trans = sample_df.values
        feat_names = sample_df.columns.tolist()

    return np.asarray(X_trans, dtype=float), feat_names


def compute_shap_explanations(
    pipeline_or_model: Any,
    X_background_raw: pd.DataFrame,
    X_explain_raw: Optional[pd.DataFrame] = None,
    problem_type: str = "classification",
    max_background: int = 60,
    max_explain: int = 60
) -> Dict[str, Any]:
    """
    Compute SHAP explanations with model-appropriate explainer and defensive fallbacks:
    - TreeExplainer for tree ensembles (RandomForest, ExtraTrees, GradientBoosting)
    - LinearExplainer for linear models (LogisticRegression, Ridge, Lasso)
    - Graceful fallback on failure or incompatibility.
    """
    if not _has_shap():
        return {
            "status": "unavailable",
            "is_available": False,
            "message": "SHAP library is not installed in the current environment.",
            "explainer_type": None
        }

    shap = _ensure_shap()
    raw_model = get_underlying_model(pipeline_or_model)
    model_info = inspect_model_family(raw_model)

    try:
        # Prepare background data
        X_bg_trans, feat_names = prepare_data_for_shap(
            pipeline_or_model, X_background_raw, max_samples=max_background
        )

        # Prepare explain data (default to subset of background if not specified)
        if X_explain_raw is not None:
            X_exp_trans, _ = prepare_data_for_shap(
                pipeline_or_model, X_explain_raw, max_samples=max_explain
            )
        else:
            X_exp_trans = X_bg_trans[:min(len(X_bg_trans), max_explain)]

        explainer = None
        explainer_type = model_info.get("shap_explainer_type")

        # Select appropriate explainer
        if explainer_type == "TreeExplainer":
            try:
                explainer = shap.TreeExplainer(raw_model)
                shap_values_raw = explainer.shap_values(X_exp_trans)
            except Exception:
                # HistGradientBoosting or unsupported tree format fallback
                masker = shap.maskers.Independent(data=X_bg_trans)
                predict_fn = getattr(raw_model, "predict_proba", raw_model.predict)
                explainer = shap.Explainer(predict_fn, masker)
                shap_values_raw = explainer(X_exp_trans)

        elif explainer_type == "LinearExplainer":
            try:
                masker = shap.maskers.Independent(data=X_bg_trans)
                explainer = shap.LinearExplainer(raw_model, masker)
                shap_values_raw = explainer.shap_values(X_exp_trans)
            except Exception:
                explainer = shap.LinearExplainer(raw_model, X_bg_trans)
                shap_values_raw = explainer.shap_values(X_exp_trans)

        else:
            # Generic model Explainer / Permutation Explainer
            sample_bg = shap.sample(X_bg_trans, min(25, len(X_bg_trans)))
            predict_fn = getattr(raw_model, "predict_proba", raw_model.predict)
            explainer = shap.Explainer(predict_fn, sample_bg)
            shap_values_raw = explainer(X_exp_trans)

        # Extract values matrix and expected base value
        base_value = None
        if hasattr(explainer, "expected_value"):
            base_value = explainer.expected_value
        elif hasattr(shap_values_raw, "base_values"):
            base_value = np.mean(shap_values_raw.base_values)

        # Normalize shap_values array
        if hasattr(shap_values_raw, "values"):
            values_mat = np.asarray(shap_values_raw.values)
        elif isinstance(shap_values_raw, list):
            # Multiclass list of arrays: average absolute values across classes
            values_mat = np.mean([np.abs(arr) for arr in shap_values_raw], axis=0)
        else:
            values_mat = np.asarray(shap_values_raw)

        # If 3D array (samples, features, classes), average across classes
        if values_mat.ndim == 3:
            values_mat = np.mean(np.abs(values_mat), axis=2)

        # Compute global feature importance via mean(|SHAP|)
        mean_abs_shap = np.mean(np.abs(values_mat), axis=0)
        if len(mean_abs_shap) == len(feat_names):
            names = feat_names
        else:
            names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]

        df_global = pd.DataFrame({
            "Feature": names,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        total = df_global["Mean_Abs_SHAP"].sum()
        df_global["Relative_Weight_%"] = (df_global["Mean_Abs_SHAP"] / total * 100.0) if total > 0 else 0.0

        return {
            "status": "success",
            "is_available": True,
            "explainer_type": explainer.__class__.__name__,
            "shap_values": values_mat,
            "X_trans": X_exp_trans,
            "feature_names": names,
            "base_value": float(np.mean(base_value)) if base_value is not None else 0.0,
            "global_importance_df": df_global,
            "top_features": df_global["Feature"].head(10).tolist(),
            "raw_explainer": explainer
        }

    except Exception as e:
        return {
            "status": "error",
            "is_available": False,
            "message": f"SHAP explanation failed gracefully: {str(e)}",
            "explainer_type": model_info.get("shap_explainer_type")
        }


# =====================================================================
# LOCAL INDIVIDUAL PREDICTION EXPLANATION
# =====================================================================

def explain_single_prediction(
    pipeline_or_model: Any,
    single_row_df: pd.DataFrame,
    X_background_raw: Optional[pd.DataFrame] = None,
    problem_type: str = "classification",
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Generate an individual prediction explanation answering:
    "Why did the model make this prediction?"
    Highlights the most influential positive and negative features.
    Explicitly clarifies correlation over causation.
    """
    raw_model = get_underlying_model(pipeline_or_model)
    preprocessor = get_preprocessor(pipeline_or_model)
    model_info = inspect_model_family(raw_model)

    # 1. Obtain model prediction
    if hasattr(pipeline_or_model, "predict"):
        prediction = pipeline_or_model.predict(single_row_df)[0]
    else:
        prediction = raw_model.predict(single_row_df)[0]

    probabilities = None
    if problem_type == "classification" and hasattr(pipeline_or_model, "predict_proba"):
        try:
            proba_arr = pipeline_or_model.predict_proba(single_row_df)[0]
            classes = getattr(pipeline_or_model, "classes_", getattr(raw_model, "classes_", None))
            if classes is not None and len(classes) == len(proba_arr):
                probabilities = {str(c): float(p) for c, p in zip(classes, proba_arr)}
        except Exception:
            probabilities = None

    # 2. Try SHAP local attribution
    shap_success = False
    contributions = []
    base_val = 0.0

    if _has_shap() and X_background_raw is not None and len(X_background_raw) >= 5:
        try:
            shap_res = compute_shap_explanations(
                pipeline_or_model,
                X_background_raw=X_background_raw,
                X_explain_raw=single_row_df,
                problem_type=problem_type,
                max_background=40,
                max_explain=1
            )
            if shap_res["is_available"] and "shap_values" in shap_res:
                s_vals = shap_res["shap_values"][0]
                f_names = shap_res["feature_names"]
                base_val = shap_res["base_value"]

                # Extract single row transformed values for feature value display
                x_trans = shap_res["X_trans"][0]

                for name, val, f_val in zip(f_names, s_vals, x_trans):
                    contributions.append({
                        "Feature": name,
                        "Contribution": float(val),
                        "Abs_Contribution": abs(float(val)),
                        "Direction": "Positive (+)" if val >= 0 else "Negative (-)",
                        "Transformed_Value": float(f_val)
                    })
                shap_success = True
        except Exception:
            shap_success = False

    # 3. Fallback Local Explanation when SHAP is unavailable
    if not shap_success:
        # Fallback: attribute using global feature importances and single row deviation
        global_imp = compute_global_feature_importance(
            pipeline_or_model,
            X_sample=X_background_raw,
            y_sample=None,
            problem_type=problem_type
        )
        if global_imp["status"] == "success" and not global_imp["importance_df"].empty:
            imp_df = global_imp["importance_df"]
            for _, r in imp_df.iterrows():
                f_name = r["Feature"]
                raw_val = single_row_df[f_name].iloc[0] if f_name in single_row_df.columns else 1.0
                try:
                    num_val = float(raw_val)
                    sign = 1.0 if num_val >= 0 else -1.0
                except (ValueError, TypeError):
                    sign = 1.0
                contrib = float(r["Importance"]) * sign
                contributions.append({
                    "Feature": f_name,
                    "Contribution": contrib,
                    "Abs_Contribution": abs(contrib),
                    "Direction": "Positive (+)" if contrib >= 0 else "Negative (-)",
                    "Transformed_Value": raw_val
                })
        else:
            # Simple uniform baseline
            cols = single_row_df.columns.tolist()
            for col in cols:
                contributions.append({
                    "Feature": col,
                    "Contribution": 1.0 / max(1, len(cols)),
                    "Abs_Contribution": 1.0 / max(1, len(cols)),
                    "Direction": "Positive (+)",
                    "Transformed_Value": single_row_df[col].iloc[0]
                })

    df_contrib = pd.DataFrame(contributions).sort_values(
        by="Abs_Contribution", ascending=False
    ).reset_index(drop=True)

    top_features_df = df_contrib.head(top_k)

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "has_probabilities": probabilities is not None,
        "base_value": base_val,
        "contributions_df": df_contrib,
        "top_contributions_df": top_features_df,
        "method": "SHAP (Local Attribution)" if shap_success else "Feature Weight Approximation (Fallback)",
        "model_type": model_info["model_name"],
        "disclaimer": (
            "These features contributed most to this model prediction. "
            "Feature attribution indicates model dependency, not real-world causation."
        )
    }


# =====================================================================
# PLOTLY INTERACTIVE VISUALIZATIONS
# =====================================================================

from modules.ui_theme import get_plotly_layout, get_tokens, get_active_theme


def plot_global_importance(
    importance_df: pd.DataFrame,
    title: str = "Global Feature Importance",
    top_n: int = 15,
    theme: Optional[str] = None
) -> go.Figure:
    """
    Interactive horizontal bar chart of global feature importance.
    """
    plot_df = importance_df.head(top_n).iloc[::-1]  # Top feature at the top
    layout = get_plotly_layout(theme)

    val_col = "Relative_Weight_%" if "Relative_Weight_%" in plot_df.columns else "Importance"
    hover_fmt = "%{x:.2f}%" if "%" in val_col else "%{x:.4f}"

    fig = go.Figure(go.Bar(
        x=plot_df[val_col],
        y=plot_df["Feature"],
        orientation="h",
        marker=dict(
            color=plot_df[val_col],
            colorscale=[[0, "#214665"], [0.5, "#39799C"], [1, "#77ADBF"]],
            showscale=False
        ),
        hovertemplate=f"<b>%{{y}}</b><br>Importance: {hover_fmt}<extra></extra>"
    ))

    fig.update_layout(
        **layout,
        title=dict(text=title, font=dict(size=14, weight=700)),
        xaxis_title="Relative Weight (%)" if "%" in val_col else "Importance Score",
        yaxis_title="Feature",
        height=max(320, len(plot_df) * 28 + 80)
    )
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    X_trans: Optional[np.ndarray] = None,
    max_features: int = 12,
    theme: Optional[str] = None
) -> go.Figure:
    """
    Plotly beeswarm/summary scatter visualization of SHAP values.
    Shows the magnitude and direction of feature effects across samples.
    """
    layout = get_plotly_layout(theme)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:max_features]

    fig = go.Figure()

    for idx in reversed(top_indices):
        f_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        vals = shap_values[:, idx]

        # Feature value coloring (normalized 0-1)
        if X_trans is not None and idx < X_trans.shape[1]:
            raw_f = X_trans[:, idx]
            rng = np.ptp(raw_f)
            color_vals = (raw_f - np.min(raw_f)) / rng if rng > 0 else np.zeros_like(raw_f)
        else:
            color_vals = np.abs(vals)

        jitter = np.random.normal(0, 0.08, size=len(vals))
        y_pos = np.full(len(vals), f_name, dtype=object)

        fig.add_trace(go.Scatter(
            x=vals,
            y=y_pos,
            mode="markers",
            name=f_name,
            showlegend=False,
            marker=dict(
                size=7,
                color=color_vals,
                colorscale=[[0, "#214665"], [0.5, "#77ADBF"], [1, "#BCD6D9"]],
                opacity=0.8,
                line=dict(width=0)
            ),
            hovertemplate=f"<b>{f_name}</b><br>SHAP Impact: %{{x:.4f}}<extra></extra>"
        ))

    fig.update_layout(
        **layout,
        title=dict(text="SHAP Feature Impact Distribution (Summary Beeswarm)", font=dict(size=14, weight=700)),
        xaxis_title="SHAP Value (Impact on Model Prediction)",
        yaxis_title="Feature",
        height=max(360, len(top_indices) * 32 + 90)
    )
    return fig


def plot_prediction_waterfall(
    contributions_df: pd.DataFrame,
    base_value: float = 0.0,
    prediction_val: Any = None,
    title: str = "Feature Contributions for this Prediction",
    theme: Optional[str] = None
) -> go.Figure:
    """
    Diverging waterfall bar chart showing individual feature contributions to a single prediction.
    """
    layout = get_plotly_layout(theme)
    tok = get_tokens(theme)
    plot_df = contributions_df.head(10).iloc[::-1]

    colors = [tok["success"] if c >= 0 else tok["error"] for c in plot_df["Contribution"]]

    fig = go.Figure(go.Bar(
        x=plot_df["Contribution"],
        y=plot_df["Feature"],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{c:+.3f}" for c in plot_df["Contribution"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.4f}<extra></extra>"
    ))

    fig.add_vline(x=0, line_width=1.5, line_color=tok["accent_primary"], line_dash="dash")

    fig.update_layout(
        **layout,
        title=dict(text=title, font=dict(size=14, weight=700)),
        xaxis_title="Attribution to Prediction (+ Supports / - Opposes)",
        yaxis_title="Feature",
        height=max(300, len(plot_df) * 32 + 80)
    )
    return fig


# =====================================================================
# STREAMLIT UI COMPONENT: MODEL EXPLANATION
# =====================================================================

def render_model_explanation_ui(
    pipeline_or_model: Any,
    model_name: str,
    problem_type: str = "classification",
    X_sample: Optional[pd.DataFrame] = None,
    y_sample: Optional[Any] = None,
    X_background: Optional[pd.DataFrame] = None
) -> None:
    """
    Renders the professional '## Model Explanation' UI section:
    - Model family and architecture inspection
    - Global feature importance (native or permutation fallback)
    - SHAP-based global explanation & beeswarm plot (on-demand)
    - Top influential features table
    - Ethical AI disclaimer distinguishing model dependency from causation
    """
    import streamlit as st

    st.markdown("## Model Explanation")
    st.markdown("<p style='color:var(--muted);margin-top:-0.3rem'>Global model behavior and feature attribution analysis.</p>", unsafe_allow_html=True)

    if pipeline_or_model is None:
        st.info("A trained model is required to generate explanations. Train a model to inspect feature importance.")
        return

    raw_model = get_underlying_model(pipeline_or_model)
    model_info = inspect_model_family(raw_model)

    # Model architecture card
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"**Model:** `{model_name}`")
        st.markdown(f"**Architecture:** `{model_info['model_name']}`")
    with m_col2:
        st.markdown(f"**Family:** `{model_info['family']}`")
        st.markdown(f"**Task Type:** `{problem_type.title()}`")
    with m_col3:
        st.markdown(f"**SHAP Explainer:** `{model_info.get('shap_explainer_type') or 'Generic'}`")
        nat_badge = "badge-success" if model_info["has_native_importance"] else "badge-primary"
        nat_text = "Native Mechanism" if model_info["has_native_importance"] else "Permutation Fallback"
        st.markdown(f"<span class='badge {nat_badge}'>{nat_text}</span>", unsafe_allow_html=True)

    st.markdown("---")

    cur_theme = st.session_state.get("theme", "dark")

    # Global Feature Importance Tab / Section
    tab_global, tab_shap = st.tabs(["Global Feature Importance", "SHAP Analysis"])

    with tab_global:
        st.markdown("#### Global Feature Importance")
        with st.spinner("Calculating feature importance..."):
            imp_res = compute_global_feature_importance(
                pipeline_or_model=pipeline_or_model,
                X_sample=X_sample if X_sample is not None else X_background,
                y_sample=y_sample,
                problem_type=problem_type
            )

        if imp_res["status"] == "success" and not imp_res["importance_df"].empty:
            st.caption(f"Importance Mechanism: **{imp_res['mechanism']}**")
            fig_imp = plot_global_importance(imp_res["importance_df"], title=f"Global Feature Importance ({model_name})", theme=cur_theme)
            st.plotly_chart(fig_imp, use_container_width=True)

            with st.expander("Feature Importance Data Table", expanded=False):
                st.dataframe(imp_res["importance_df"], use_container_width=True)
        else:
            st.info(imp_res.get("message", "Feature importance is not available for this model configuration."))

    with tab_shap:
        st.markdown("#### SHAP (SHapley Additive exPlanations)")
        if not _has_shap():
            st.warning("SHAP library is not installed in the environment. Feature importance fallback is active.")
        else:
            st.caption("SHAP values provide theoretically unified local and global feature attribution.")
            cache_key = f"shap_{model_name}_{id(pipeline_or_model)}"
            has_computed = cache_key in st.session_state

            c_btn, c_note = st.columns([1, 2])
            with c_btn:
                compute_btn = st.button("Generate SHAP Visualizations", key=f"btn_shap_{model_name}")
            with c_note:
                st.markdown("<span style='color:var(--muted);font-size:0.8rem'>Computes SHAP values over sampled background rows for fast, responsive rendering.</span>", unsafe_allow_html=True)

            if compute_btn or has_computed:
                if compute_btn or not has_computed:
                    with st.spinner("Computing SHAP values (using optimized background sampling)..."):
                        bg_data = X_background if X_background is not None else X_sample
                        shap_result = compute_shap_explanations(
                            pipeline_or_model=pipeline_or_model,
                            X_background_raw=bg_data,
                            X_explain_raw=X_sample,
                            problem_type=problem_type,
                            max_background=50,
                            max_explain=50
                        )
                        st.session_state[cache_key] = shap_result
                else:
                    shap_result = st.session_state[cache_key]

                if shap_result.get("is_available", False):
                    st.success(f"Computed with **{shap_result.get('explainer_type', 'Explainer')}**")
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        # Beeswarm summary plot
                        fig_beeswarm = plot_shap_summary(
                            shap_values=shap_result["shap_values"],
                            feature_names=shap_result["feature_names"],
                            X_trans=shap_result.get("X_trans"),
                            theme=cur_theme
                        )
                        st.plotly_chart(fig_beeswarm, use_container_width=True)
                    with sc2:
                        # Global mean(|SHAP|) bar plot
                        fig_bar = plot_global_importance(
                            shap_result["global_importance_df"],
                            title="Mean |SHAP Value| (Global Impact)",
                            theme=cur_theme
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning(shap_result.get("message", "SHAP explanation could not be computed for this model architecture."))

    st.markdown("""<div class="ai-card" style="font-size:0.8rem;color:var(--muted);margin-top:1rem">
    <strong>Interpretation Note:</strong> These features contributed most to this model's predictions.
    Feature attribution measures mathematical dependency within the trained model and does not prove real-world causality.
    </div>""", unsafe_allow_html=True)

