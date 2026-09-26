"""
MLForge - Dynamic Prediction Playground Engine
====================================================
Provides an end-to-end interactive prediction interface:
- Extracts feature schemas directly from raw training datasets (numeric, categorical, boolean, date).
- Excludes target columns and internal transformed features.
- Defensively validates inputs (type checking, coercion, missing values, unseen categories).
- Feeds raw input dictionaries directly through the saved sklearn Pipeline (preprocessing + model).
- Outputs classification labels with class probabilities (or clear notice when probabilities are unavailable),
  and regression numerical values (never labeled as 'confidence').
- Triggers prediction explanations answering "Why did the model make this prediction?".
- Handles absent model states gracefully without crashing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from modules.explainability import (
    explain_single_prediction,
    plot_prediction_waterfall,
    inspect_model_family,
    get_underlying_model
)


# =====================================================================
# FEATURE SCHEMA EXTRACTION
# =====================================================================

def extract_feature_schema(
    df: pd.DataFrame,
    target_col: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extract predictor schema from a DataFrame, strictly omitting the target column
    and any internal transformed features.
    Returns metadata required to dynamically generate form controls:
    - type: 'numeric' | 'categorical' | 'boolean' | 'date'
    - min_val, max_val, step, default_val
    - allowed_values / unique categories
    - is_nullable
    """
    if df is None or df.empty:
        return {}

    feature_cols = [c for c in df.columns if c != target_col]
    schema: Dict[str, Dict[str, Any]] = {}

    for col in feature_cols:
        series = df[col]
        dtype = series.dtype
        n_missing = int(series.isna().sum())
        is_nullable = n_missing > 0

        # Boolean detection
        if pd.api.types.is_bool_dtype(series) or (series.dropna().isin([True, False, 0, 1]).all() and series.nunique() <= 2 and dtype == "bool"):
            schema[col] = {
                "name": col,
                "type": "boolean",
                "default": bool(series.dropna().iloc[0]) if not series.dropna().empty else False,
                "is_nullable": is_nullable
            }

        # Datetime detection
        elif pd.api.types.is_datetime64_any_dtype(series):
            schema[col] = {
                "name": col,
                "type": "date",
                "min": series.min(),
                "max": series.max(),
                "default": series.dropna().iloc[0] if not series.dropna().empty else pd.Timestamp.now(),
                "is_nullable": is_nullable
            }

        # Numeric detection (integers, floats)
        elif pd.api.types.is_numeric_dtype(series):
            clean_s = series.dropna()
            if clean_s.empty:
                min_v, max_v, med_v = 0.0, 100.0, 0.0
            else:
                min_v = float(clean_s.min())
                max_v = float(clean_s.max())
                med_v = float(clean_s.median())

            # Integer or float resolution
            is_int = pd.api.types.is_integer_dtype(series) or (clean_s % 1 == 0).all()
            if is_int:
                step = 1
                min_v = int(min_v)
                max_v = int(max_v)
                med_v = int(round(med_v))
            else:
                step = round(max((max_v - min_v) / 100.0, 0.01), 4) if max_v > min_v else 0.1

            schema[col] = {
                "name": col,
                "type": "numeric",
                "is_integer": is_int,
                "min": min_v,
                "max": max_v,
                "step": step,
                "default": med_v,
                "mean": float(clean_s.mean()) if not clean_s.empty else 0.0,
                "is_nullable": is_nullable
            }

        # Categorical / String detection
        else:
            clean_s = series.dropna().astype(str)
            unique_vals = clean_s.unique().tolist()
            if not unique_vals:
                unique_vals = ["Unknown"]

            # Cap unique options in dropdown for performance
            capped_options = unique_vals[:200]
            default_val = unique_vals[0] if unique_vals else ""

            schema[col] = {
                "name": col,
                "type": "categorical",
                "options": capped_options,
                "default": default_val,
                "total_categories": len(unique_vals),
                "is_nullable": is_nullable
            }

    return schema


# =====================================================================
# DEFENSIVE INPUT VALIDATION & SANITIZATION
# =====================================================================

def validate_and_sanitize_inputs(
    raw_inputs: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]]
) -> Tuple[bool, Optional[pd.DataFrame], List[str], List[str]]:
    """
    Validate user form inputs against the expected feature schema:
    - Type coercion and checks (numeric, categorical, boolean)
    - Handling missing values according to schema
    - Detecting unknown categories and allowing pipeline OneHotEncoder(handle_unknown='ignore') to handle them
    - Catching missing required fields
    Returns:
        is_valid: bool
        sanitized_df: pd.DataFrame (1-row) or None
        errors: List[str]
        warnings: List[str]
    """
    errors: List[str] = []
    warnings: List[str] = []
    sanitized_row: Dict[str, Any] = {}

    if not schema:
        return False, None, ["Model feature schema is empty or unavailable."], []

    for feat_name, meta in schema.items():
        feat_type = meta["type"]

        # Check field presence
        if feat_name not in raw_inputs:
            # If omitted, default or mark missing
            if meta.get("is_nullable", False):
                sanitized_row[feat_name] = np.nan
                warnings.append(f"Field '{feat_name}' omitted: imputed as NaN.")
            else:
                sanitized_row[feat_name] = meta.get("default", 0)
                warnings.append(f"Field '{feat_name}' was not provided. Using default: {meta.get('default')}.")
            continue

        raw_val = raw_inputs[feat_name]

        # Handle empty strings or None
        if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
            if meta.get("is_nullable", False):
                sanitized_row[feat_name] = np.nan
                warnings.append(f"Field '{feat_name}' is empty: imputed as missing (NaN).")
            else:
                # Use default
                sanitized_row[feat_name] = meta.get("default", 0)
                warnings.append(f"Field '{feat_name}' is empty. Replaced with default: {meta.get('default')}.")
            continue

        # Type validation & coercion
        if feat_type == "numeric":
            try:
                # Handle numeric conversion
                coerced = float(raw_val)
                if np.isnan(coerced) or np.isinf(coerced):
                    sanitized_row[feat_name] = np.nan
                elif meta.get("is_integer", False):
                    sanitized_row[feat_name] = int(round(coerced))
                else:
                    sanitized_row[feat_name] = coerced
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric input for '{feat_name}': expected number, received '{raw_val}'.")

        elif feat_type == "categorical":
            str_val = str(raw_val).strip()
            # Check if category is unseen
            known_opts = meta.get("options", [])
            if known_opts and str_val not in known_opts:
                warnings.append(f"Category '{str_val}' for feature '{feat_name}' was not seen in training data. Pipeline will safely ignore it.")
            sanitized_row[feat_name] = str_val

        elif feat_type == "boolean":
            if isinstance(raw_val, bool):
                sanitized_row[feat_name] = raw_val
            elif str(raw_val).lower() in ("true", "1", "yes"):
                sanitized_row[feat_name] = True
            elif str(raw_val).lower() in ("false", "0", "no"):
                sanitized_row[feat_name] = False
            else:
                errors.append(f"Invalid boolean value for '{feat_name}': received '{raw_val}'.")

        elif feat_type == "date":
            try:
                sanitized_row[feat_name] = pd.to_datetime(raw_val)
            except Exception:
                errors.append(f"Invalid date format for '{feat_name}': '{raw_val}'.")

        else:
            sanitized_row[feat_name] = raw_val

    if errors:
        return False, None, errors, warnings

    df_out = pd.DataFrame([sanitized_row])
    # Ensure correct column ordering matching schema
    ordered_cols = [c for c in schema.keys() if c in df_out.columns]
    df_out = df_out[ordered_cols]

    return True, df_out, errors, warnings


# =====================================================================
# PIPELINE PREDICTION EXECUTION
# =====================================================================

def execute_pipeline_prediction(
    pipeline: Any,
    raw_input_df: pd.DataFrame,
    problem_type: str = "classification"
) -> Dict[str, Any]:
    """
    Run raw input DataFrame through the saved end-to-end Pipeline.
    Architecture:
        Raw User Input -> Preprocessing + Model Pipeline -> Prediction Output
    Never recreates preprocessing manually; uses pipeline.predict() directly.
    """
    if pipeline is None:
        return {
            "status": "error",
            "message": "No trained model pipeline provided."
        }

    try:
        raw_model = get_underlying_model(pipeline)
        model_info = inspect_model_family(raw_model)

        # 1. Prediction execution
        preds = pipeline.predict(raw_input_df)
        pred_val = preds[0]

        # 2. Probability extraction (classification only)
        probabilities: Optional[Dict[str, float]] = None
        has_probabilities = False

        if problem_type == "classification":
            if hasattr(pipeline, "predict_proba"):
                try:
                    proba = pipeline.predict_proba(raw_input_df)[0]
                    # Map classes
                    classes = getattr(pipeline, "classes_", getattr(raw_model, "classes_", None))
                    if classes is not None and len(classes) == len(proba):
                        probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
                        has_probabilities = True
                except Exception:
                    probabilities = None
                    has_probabilities = False
        else:
            # Regression: ensure pred_val is numeric float
            try:
                pred_val = float(pred_val)
            except Exception:
                pass

        return {
            "status": "success",
            "prediction": pred_val,
            "probabilities": probabilities,
            "has_probabilities": has_probabilities,
            "problem_type": problem_type,
            "model_type": model_info["model_name"],
            "model_family": model_info["family"]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Pipeline prediction execution failed: {str(e)}"
        }


# =====================================================================
# FULL PLAYGROUND WORKFLOW ORCHESTRATION
# =====================================================================

def run_prediction_playground(
    pipeline: Any,
    raw_inputs: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]],
    problem_type: str = "classification",
    X_background: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    End-to-end orchestration:
    1. Validates & sanitizes raw inputs against feature schema.
    2. Runs sanitized DataFrame through the fitted pipeline.
    3. Generates prediction explanation (SHAP or fallback).
    """
    # Check model state
    if pipeline is None:
        return {
            "status": "no_model",
            "message": "No compatible trained model exists. Please train a model first in Step 7."
        }

    # 1. Validate inputs
    is_valid, sanitized_df, errors, warnings = validate_and_sanitize_inputs(raw_inputs, schema)
    if not is_valid or sanitized_df is None:
        return {
            "status": "validation_error",
            "errors": errors,
            "warnings": warnings
        }

    # 2. Execute pipeline prediction
    pred_res = execute_pipeline_prediction(pipeline, sanitized_df, problem_type=problem_type)
    if pred_res["status"] != "success":
        return {
            "status": "prediction_error",
            "message": pred_res.get("message", "Prediction failed."),
            "warnings": warnings
        }

    # 3. Generate individual prediction explanation
    explanation = explain_single_prediction(
        pipeline_or_model=pipeline,
        single_row_df=sanitized_df,
        X_background_raw=X_background,
        problem_type=problem_type
    )

    return {
        "status": "success",
        "prediction": pred_res["prediction"],
        "probabilities": pred_res["probabilities"],
        "has_probabilities": pred_res["has_probabilities"],
        "problem_type": problem_type,
        "model_type": pred_res["model_type"],
        "model_family": pred_res["model_family"],
        "warnings": warnings,
        "explanation": explanation,
        "sanitized_input": sanitized_df.iloc[0].to_dict()
    }


# =====================================================================
# STREAMLIT UI COMPONENT: PREDICTION PLAYGROUND
# =====================================================================

def render_prediction_playground_ui(
    trained_models: Dict[str, Any],
    default_model_name: Optional[str] = None,
    problem_type: str = "classification",
    feature_schema: Optional[Dict[str, Dict[str, Any]]] = None,
    X_background: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None
) -> None:
    """
    Renders the professional '## Prediction Playground' and '## Prediction Explanation' UI sections.
    - Gracefully notifies the user when no model has been trained.
    - Dynamically generates schema-driven input controls (numeric, categorical, boolean, date).
    - Prevents target column or transformed features from being exposed as inputs.
    - Executes prediction through the saved end-to-end pipeline.
    - Displays prediction outputs (class + probabilities for classification, value for regression).
    - Renders local prediction explanations with ethical AI disclaimers.
    """
    import streamlit as st
    import plotly.graph_objects as go
    from modules.ui_theme import get_plotly_layout, get_tokens, render_empty_state

    st.markdown("## Prediction Playground")
    st.markdown("<p style='color:var(--muted);margin-top:-0.3rem'>Interactive user prediction using the fitted end-to-end preprocessing and model pipeline.</p>", unsafe_allow_html=True)

    # 1. Check Model State
    if not trained_models:
        render_empty_state(
            title="No Trained Pipeline Available",
            description="A trained machine learning pipeline is required to perform interactive inference and feature attribution. Please navigate to Step 7 and train a model first.",
            cta_label="Go to Model Training",
            cta_step=7
        )
        return

    # Model selector
    model_names = list(trained_models.keys())
    idx = model_names.index(default_model_name) if default_model_name in model_names else 0

    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        chosen_model_name = st.selectbox("Active Pipeline for Inference", model_names, index=idx, key="playground_model_select")
    with c_sel2:
        st.markdown(f"<div style='margin-top:1.8rem'><span class='badge badge-primary'>{problem_type.title()}</span></div>", unsafe_allow_html=True)

    pipeline = trained_models[chosen_model_name]

    # Resolve schema
    schema = feature_schema
    if not schema and hasattr(pipeline, "feature_schema_"):
        schema = pipeline.feature_schema_
    if not schema and X_background is not None:
        schema = extract_feature_schema(X_background, target_col=target_col)

    if not schema:
        st.warning("Feature schema could not be derived. Please ensure training data is available.")
        return

    st.markdown("#### Feature Input Controls")
    st.markdown("<p style='color:var(--muted);font-size:0.8rem'>Provide raw feature values below. Values will be processed automatically by the pipeline's fitted ColumnTransformer.</p>", unsafe_allow_html=True)

    # Generate Dynamic Form Inputs
    raw_inputs: Dict[str, Any] = {}
    schema_items = list(schema.items())

    # Lay out fields in 3 columns
    cols_per_row = 3
    num_items = len(schema_items)

    for i in range(0, num_items, cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            item_idx = i + j
            if item_idx < num_items:
                col_name, meta = schema_items[item_idx]
                f_type = meta["type"]
                field_key = f"pg_input_{col_name}"

                with row_cols[j]:
                    if f_type == "numeric":
                        min_v = float(meta.get("min", 0.0))
                        max_v = float(meta.get("max", 1000000.0))
                        def_v = float(meta.get("default", 0.0))
                        step_v = float(meta.get("step", 0.1))

                        # Guard bounds
                        if min_v > max_v:
                            min_v, max_v = max_v, min_v
                        if def_v < min_v or def_v > max_v:
                            def_v = min_v

                        if meta.get("is_integer", False):
                            raw_inputs[col_name] = st.number_input(
                                label=col_name,
                                min_value=int(min_v),
                                max_value=int(max_v),
                                value=int(def_v),
                                step=int(step_v) if step_v >= 1 else 1,
                                key=field_key
                            )
                        else:
                            raw_inputs[col_name] = st.number_input(
                                label=col_name,
                                min_value=min_v,
                                max_value=max_v,
                                value=def_v,
                                step=step_v,
                                key=field_key
                            )

                    elif f_type == "categorical":
                        opts = meta.get("options", ["Standard"])
                        def_opt = meta.get("default", opts[0])
                        opt_idx = opts.index(def_opt) if def_opt in opts else 0
                        raw_inputs[col_name] = st.selectbox(
                            label=col_name,
                            options=opts,
                            index=opt_idx,
                            key=field_key
                        )

                    elif f_type == "boolean":
                        def_bool = bool(meta.get("default", False))
                        raw_inputs[col_name] = st.checkbox(
                            label=col_name,
                            value=def_bool,
                            key=field_key
                        )

                    elif f_type == "date":
                        raw_inputs[col_name] = st.date_input(
                            label=col_name,
                            key=field_key
                        )

                    else:
                        raw_inputs[col_name] = st.text_input(
                            label=col_name,
                            value=str(meta.get("default", "")),
                            key=field_key
                        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Generate Prediction", key="btn_run_playground_prediction", use_container_width=True)

    if predict_clicked:
        with st.spinner("Processing input through pipeline..."):
            play_result = run_prediction_playground(
                pipeline=pipeline,
                raw_inputs=raw_inputs,
                schema=schema,
                problem_type=problem_type,
                X_background=X_background
            )

        if play_result["status"] == "validation_error":
            for err in play_result.get("errors", []):
                st.error(f"Input Validation Error: {err}")
            return

        if play_result["status"] == "prediction_error":
            st.error(f"Prediction Error: {play_result.get('message', 'Inference failed.')}")
            return

        for warn in play_result.get("warnings", []):
            st.warning(f"Note: {warn}")

        # PREDICTION OUTPUT SECTION
        st.markdown("---")
        st.markdown("### Prediction Output")

        pred_val = play_result["prediction"]
        has_proba = play_result.get("has_probabilities", False)
        probas = play_result.get("probabilities")

        if problem_type == "classification":
            out_c1, out_c2 = st.columns([1, 2])
            with out_c1:
                st.markdown(f"""<div class="metric-card" style="margin-top:0.5rem">
                <div class="metric-label">Predicted Class</div>
                <div class="metric-value" style="font-size:1.8rem;color:var(--accent)">{pred_val}</div>
                </div>""", unsafe_allow_html=True)

            with out_c2:
                if has_proba and probas:
                    st.markdown("#### Class Probabilities")
                    prob_df = pd.DataFrame(list(probas.items()), columns=["Class", "Probability"]).sort_values(by="Probability", ascending=True)

                    cur_theme = st.session_state.get("theme", "dark")
                    layout_p = get_plotly_layout(cur_theme)

                    fig_prob = go.Figure(go.Bar(
                        x=prob_df["Probability"],
                        y=prob_df["Class"].astype(str),
                        orientation="h",
                        marker=dict(
                            color=prob_df["Probability"],
                            colorscale=[[0, "#214665"], [0.5, "#39799C"], [1, "#77ADBF"]]
                        ),
                        text=[f"{p:.1%}" for p in prob_df["Probability"]],
                        textposition="outside"
                    ))
                    fig_prob.update_layout(
                        **layout_p,
                        xaxis=dict(range=[0, 1.15], gridcolor=layout_p["xaxis"]["gridcolor"]),
                        height=max(200, len(prob_df) * 35 + 50)
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                else:
                    st.info("Class probabilities are not available for this model configuration (e.g. non-probabilistic decision boundary).")

        else:
            # Regression: Never call it "confidence"
            st.markdown(f"""<div class="metric-card metric-card-primary" style="margin-top:0.5rem;max-width:350px">
            <div class="metric-label">Predicted Value (Continuous)</div>
            <div class="metric-value">{float(pred_val):.4f}</div>
            </div>""", unsafe_allow_html=True)

        # PREDICTION EXPLANATION SECTION
        st.markdown("---")
        st.markdown("## Prediction Explanation")
        st.markdown("<p style='color:var(--muted);margin-top:-0.3rem'>Why did the model make this prediction?</p>", unsafe_allow_html=True)

        explanation = play_result.get("explanation", {})
        contrib_df = explanation.get("contributions_df")

        if contrib_df is not None and not contrib_df.empty:
            st.caption(f"Explanation Engine: **{explanation.get('method', 'Local Attribution')}**")

            # Diverging waterfall chart
            cur_theme = st.session_state.get("theme", "dark")
            fig_waterfall = plot_prediction_waterfall(
                contributions_df=contrib_df,
                prediction_val=pred_val,
                title=f"Feature Contributions towards Prediction ({chosen_model_name})",
                theme=cur_theme
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

            # Top contributors summary table
            with st.expander("Detailed Feature Contribution Breakdown", expanded=False):
                st.dataframe(contrib_df, use_container_width=True)

        st.markdown("""<div class="ai-card" style="font-size:0.8rem;color:var(--muted);margin-top:1rem">
        <strong>Attribution Disclaimer:</strong> These features contributed most to this model prediction.
        Feature attribution describes model dependency and internal weights; it does not prove real-world causation.
        </div>""", unsafe_allow_html=True)

