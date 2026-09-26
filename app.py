import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os, io, time, traceback

sys.path.insert(0, os.path.dirname(__file__))

favicon_path = os.path.join(os.path.dirname(__file__), "assets", "branding", "favicon", "favicon-32.png")
st.set_page_config(
    page_title="MLForge",
    page_icon=favicon_path if os.path.exists(favicon_path) else "M",
    layout="wide",
    initial_sidebar_state="expanded"
)

from modules.ui_theme import (
    inject_custom_css, get_plotly_layout, render_platform_header,
    render_dataset_kpi_bar, render_step_header, render_empty_state,
    render_structured_error, get_tokens, get_active_theme, get_brand_symbol_svg,
    render_step_navigation
)

# Apply Centralized Enterprise SaaS Stylesheet
inject_custom_css()

# Startup Splash Screen (Runs once per session)
from modules.splash_screen import show_splash_screen

qp = st.query_params
force_splash = qp.get("splash_demo") in ["1", "true", "True"] or qp.get("splash") in ["1", "true", "True"]
if force_splash and not st.session_state.get("_splash_forced_once", False):
    st.session_state.splash_completed = False
    st.session_state._splash_forced_once = True

if not st.session_state.get("splash_completed", False):
    show_splash_screen(duration_sec=3.8, force=True)
    time.sleep(3.8)
    st.session_state.splash_completed = True
    st.rerun()



#  Module imports 
from modules.file_loader import load_file
from modules.profiling import get_basic_info, get_column_summary, get_numeric_stats, get_categorical_stats
from modules.missing_handler import get_missing_summary, fill_missing_values, suggest_missing_strategy
from modules.duplicate_handler import remove_duplicates
from modules.exporter import export_data
from modules.clustering import (prepare_clustering_data, run_kmeans, run_dbscan,
                                run_agglomerative, reduce_to_2d, find_optimal_clusters,
                                get_best_clustering, run_all_clustering)
from modules.ai_recommender import recommend_clustering, generate_ai_report, recommend_model_export
from modules.data_audit import (run_data_audit, render_audit_summary_cards,
                                render_audit_full_dashboard, render_target_preflight_check)
from modules.automl import (detect_problem_type, prepare_train_test_split,
                            train_and_evaluate_models)
from modules.evaluation import get_available_metrics, is_higher_better, METRIC_CONFIGS
from modules.model_export.export_manager import export_model
from modules.explainability import render_model_explanation_ui
from modules.prediction_playground import render_prediction_playground_ui, extract_feature_schema
from utils.icons import hgi, nav_icon, status_icon, card_icon, header_icon, inline_icon, icon_label, ICON_SIZE

#  Session State ─
DEFAULTS = dict(df_raw=None, df_clean=None, df_cleaned_only=None, step=1,
                theme="dark", splash_completed=True,
                trained_models={}, results_df=None, best_model_name=None,
                problem_type=None, target_col=None, cluster_results=None,
                X_train=None, y_train=None, X_test=None, y_test=None, ai_report=None,
                feature_schema={}, shap_cache={},
                missing_applied=False, dup_removed=False,
                preprocessing_log=[], training_dataset="preprocessed",
                audit_result=None, audit_target_result=None,
                optimization_metric=None, cv_folds=5, test_size_pct=20)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

#  Helpers ─
PT = get_plotly_layout()
COLORS = ["#e78a53", "#5f8787", "#fbcb97", "#888888", "#10b981", "#f59e0b", "#ef4444", "#999999"]

def mc(val, label, variant="primary"):
    return f'<div class="metric-card metric-card-{variant}"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>'

def sh(icon, title, sub=""):
    render_step_header(st.session_state.step, title, sub)

def footer():
    st.markdown("""<div class="footer-bar">
    <span>MLForge &nbsp;&bull;&nbsp; AutoML Workspace</span>
    </div>""", unsafe_allow_html=True)

def detect_problem_type(df, target_col):
    if target_col not in df.columns: return "classification"
    y = df[target_col].dropna()
    if y.empty: return "classification"  # guard: all-null column
    if y.dtype == "object": return "classification"
    if pd.api.types.is_numeric_dtype(y) and y.nunique() < max(10, int(0.05*len(y))):
        return "classification"
    return "regression"


# SIDEBAR

with st.sidebar:
    brand_icon = get_brand_symbol_svg(34, "dark")
    st.markdown(f"""<div class="sidebar-brand-header" style="padding:1.45rem 0.2rem 0.65rem;border-bottom:1px solid var(--mf-border);margin-bottom:0.6rem;display:flex;align-items:center;gap:0.65rem">
        {brand_icon}
        <span class="sidebar-brand-text" style="font-family:'Geist Sans',sans-serif;font-size:1.45rem;font-weight:700;letter-spacing:-0.03em;color:var(--mf-text-primary);line-height:1">MLForge</span>
    </div>""", unsafe_allow_html=True)

    # Step → (label, icon_slug)
    NAV_GROUPS = [
        ("Workspace", [
            (1,  "Data Ingestion",       "database-import"),
            (2,  "Profiling & Health",   "analytics-01"),
            (3,  "Missing Values",       "filter-remove"),
            (4,  "Duplicate Records",    "layers-01"),
            (5,  "Preprocessing",        "sliders-horizontal"),
        ]),
        ("Modeling", [
            (7,  "Model Training",       "neural-network"),
            (8,  "Cluster Visualizer",   "chart-scatter"),
            (11, "Explainability",       "eye"),
            (10, "Prediction Playground","target-01"),
        ]),
        ("Outputs", [
            (9,  "Reports",              "analytics-02"),
            (6,  "Export Center",        "file-export"),
        ])
    ]

    has_data = st.session_state.df_raw is not None
    current_step = st.session_state.step

    for group_title, steps in NAV_GROUPS:
        st.markdown(
            f"<div style='font-size:0.62rem;font-weight:700;letter-spacing:0.09em;"
            f"color:var(--mf-text-muted);text-transform:uppercase;margin:0.45rem 0 0.12rem 0.2rem'>"
            f"{group_title}</div>",
            unsafe_allow_html=True
        )
        for num, label, icon_slug in steps:
            is_active = (current_step == num)
            is_locked = (not has_data) and (num != 1)

            c_ic, c_bt = st.columns([1.2, 8.8])
            with c_ic:
                ic_state = "active" if is_active else ("locked" if is_locked else "")
                st.markdown(
                    f'<div class="nav-ic-box {ic_state}">'
                    f'<i class="hgi hgi-stroke hgi-{icon_slug}"></i></div>',
                    unsafe_allow_html=True
                )
            with c_bt:
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"nav_step_{num}", use_container_width=True, disabled=is_locked, type=btn_type):
                    st.session_state.step = num
                    st.rerun()

    if has_data:
        st.markdown(
            '<div style="margin-top:0.85rem;padding-top:0.65rem;border-top:1px solid var(--mf-border);'
            'font-size:0.68rem;font-weight:700;letter-spacing:0.06em;color:var(--mf-text-muted);'
            'text-transform:uppercase;margin-bottom:0.35rem;display:flex;align-items:center;gap:0.35rem">'
            '<i class="hgi hgi-stroke hgi-database-01" style="font-size:13px;color:var(--mf-text-muted)" aria-hidden="true"></i>'
            'Active Dataset</div>',
            unsafe_allow_html=True
        )
        active_df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(active_df):,}")
        c2.metric("Cols", f"{len(active_df.columns):,}")
        if st.session_state.audit_result is not None:
            ar = st.session_state.audit_result
            score = ar.score
            badge = "success" if score >= 85 else ("warning" if score >= 60 else "error")
            slug_s = "tick-02" if score >= 85 else ("alert-01" if score >= 60 else "alert-02")
            st.markdown(
                f'<div style="font-size:0.75rem;color:var(--mf-text-secondary);margin-top:0.25rem;'
                f'display:flex;align-items:center;gap:0.35rem">'
                f'<i class="hgi hgi-stroke hgi-{slug_s}" style="font-size:13px;color:var(--mf-{badge})" aria-hidden="true"></i>'
                f'Health: <strong>{score}/100</strong></div>',
                unsafe_allow_html=True
            )
        if st.session_state.target_col:
            st.markdown(
                f'<div style="font-size:0.75rem;color:var(--mf-text-secondary);margin-top:0.15rem;'
                f'display:flex;align-items:center;gap:0.35rem">'
                f'<i class="hgi hgi-stroke hgi-target-01" style="font-size:13px;color:var(--mf-text-muted)" aria-hidden="true"></i>'
                f'Target: <strong>{st.session_state.target_col}</strong></div>',
                unsafe_allow_html=True
            )
        if st.session_state.best_model_name:
            st.markdown(
                f'<div style="font-size:0.75rem;color:var(--mf-text-secondary);margin-top:0.15rem;'
                f'display:flex;align-items:center;gap:0.35rem">'
                f'<i class="hgi hgi-stroke hgi-cpu" style="font-size:13px;color:var(--mf-accent)" aria-hidden="true"></i>'
                f'Model: <strong>{st.session_state.best_model_name}</strong></div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div style="margin-top:0.75rem;padding-top:0.6rem;border-top:1px solid var(--mf-border);'
        'font-size:0.68rem;font-weight:700;letter-spacing:0.06em;color:var(--mf-text-muted);'
        'text-transform:uppercase;margin-bottom:0.25rem;display:flex;align-items:center;gap:0.3rem">'
        '<i class="hgi hgi-stroke hgi-settings-01" style="font-size:13px;color:var(--mf-text-muted)" aria-hidden="true"></i>'
        'Workspace Actions</div>',
        unsafe_allow_html=True
    )
    if st.button("Reset Workspace", key="sidebar_reset_btn", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k in DEFAULTS:
                st.session_state[k] = DEFAULTS[k]
        st.rerun()

    st.markdown("""<div style='margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid var(--mf-border);padding-bottom:1.5rem'>
        <div style='color:var(--mf-text-muted);font-size:0.66rem;text-align:center'>MLForge v2.0 &nbsp;&bull;&nbsp; Production AutoML</div>
    </div>""", unsafe_allow_html=True)



# LIVE TELEMETRY BAR

active_df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
render_dataset_kpi_bar(
    active_df,
    audit_result=st.session_state.get("audit_result"),
    target_col=st.session_state.get("target_col"),
    problem_type=st.session_state.get("problem_type"),
    best_model_name=st.session_state.get("best_model_name")
)


# STEP 1 — DATA INGESTION & BENCHMARKS

if st.session_state.step == 1:
    sh("", "Data Ingestion & Benchmark Studio", "Upload tabular datasets or initialize standard machine learning benchmarks")

    c_upload, c_benchmark = st.columns([3, 2], gap="large")
    with c_upload:
        st.markdown("""<div style="font-size:0.95rem;font-weight:700;color:var(--mf-text-primary);display:flex;align-items:center;gap:0.4rem;margin-bottom:0.25rem">
            <i class="hgi hgi-stroke hgi-upload-01" style="color:var(--mf-accent);font-size:17px"></i> Ingest Tabular Dataset
        </div>
        <div style="font-size:0.80rem;color:var(--mf-text-secondary);margin-bottom:0.6rem">
            Drop file or browse your device to begin automated profiling & ML
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop file here or click to browse",
            type=["csv", "xlsx", "xls", "json", "xml", "yaml", "yml", "db"],
            help="Supported formats: CSV, Excel, JSON, XML, YAML, SQLite DB",
            label_visibility="collapsed"
        )
        st.markdown("""<div style="display:flex;align-items:center;gap:0.35rem;margin-top:0.35rem;flex-wrap:wrap">
            <span style="font-size:0.70rem;color:var(--mf-text-muted);font-weight:600;margin-right:0.15rem">FORMATS:</span>
            <span class="badge badge-neutral">CSV</span>
            <span class="badge badge-neutral">XLSX</span>
            <span class="badge badge-neutral">JSON</span>
            <span class="badge badge-neutral">SQLITE</span>
            <span class="badge badge-neutral">YAML</span>
        </div>""", unsafe_allow_html=True)

    with c_benchmark:
        st.markdown("""<div style="font-size:0.95rem;font-weight:700;color:var(--mf-text-primary);display:flex;align-items:center;gap:0.4rem;margin-bottom:0.25rem">
            <i class="hgi hgi-stroke hgi-test-tube-01" style="color:var(--mf-accent);font-size:17px"></i> Standard Benchmarks
        </div>
        <div style="font-size:0.80rem;color:var(--mf-text-secondary);margin-bottom:0.6rem">
            Quickly test the pipeline with a pre-configured tabular benchmark
        </div>""", unsafe_allow_html=True)
        benchmark_options = {
            "Iris Flower (Classification · 150 rows · 4 features)": "iris",
            "Wine Recognition (Multiclass · 178 rows · 13 features)": "wine",
            "Breast Cancer Diagnostic (Binary · 569 rows · 30 features)": "breast_cancer",
            "Diabetes Progression (Regression · 442 rows · 10 features)": "diabetes"
        }
        selected_bench = st.selectbox(
            "Select Benchmark Dataset",
            list(benchmark_options.keys()),
            key="benchmark_selector",
            label_visibility="collapsed"
        )
        st.markdown("<div style='margin-top:0.45rem'></div>", unsafe_allow_html=True)
        if st.button("Load Benchmark Dataset", use_container_width=True, key="btn_load_benchmark", type="primary"):
            bench_id = benchmark_options[selected_bench]
            if bench_id == "iris":
                from sklearn.datasets import load_iris
                ds = load_iris(as_frame=True)
                df_b = ds.frame
                t_col = "target"
                p_type = "classification"
            elif bench_id == "wine":
                from sklearn.datasets import load_wine
                ds = load_wine(as_frame=True)
                df_b = ds.frame
                t_col = "target"
                p_type = "classification"
            elif bench_id == "breast_cancer":
                from sklearn.datasets import load_breast_cancer
                ds = load_breast_cancer(as_frame=True)
                df_b = ds.frame
                t_col = "target"
                p_type = "classification"
            else: # diabetes
                from sklearn.datasets import load_diabetes
                ds = load_diabetes(as_frame=True)
                df_b = ds.frame
                t_col = "target"
                p_type = "regression"

            st.session_state.df_raw = df_b
            st.session_state.df_clean = df_b.copy()
            st.session_state.target_col = t_col
            st.session_state.problem_type = p_type
            st.session_state.audit_result = run_data_audit(df_b)
            st.session_state.step = 2
            st.rerun()

    if uploaded:
        with st.spinner("Parsing uploaded file schema and contents..."):
            df, msg = load_file(uploaded)
        if df is not None:
            st.session_state.df_raw = df
            st.session_state.df_clean = df.copy()
            st.session_state.audit_result = run_data_audit(df)
            st.success(f"Successfully loaded **{uploaded.name}** — {len(df):,} records × {len(df.columns)} features")
        else:
            render_structured_error(
                title="File Ingestion Failed",
                what_happened=f"Could not parse the uploaded file: {uploaded.name}",
                why=msg,
                what_to_do="Ensure the file is not corrupted, has a valid header row, and matches one of the supported formats."
            )

    # If dataset is loaded, display preview and audit summary
    if st.session_state.df_clean is not None:
        st.markdown("---")
        df_cur = st.session_state.df_clean
        st.markdown("### Active Dataset Overview & Schema Preview")
        
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        c_kpi1.markdown(mc(f"{len(df_cur):,}", "Total Records"), unsafe_allow_html=True)
        c_kpi2.markdown(mc(f"{len(df_cur.columns)}", "Features"), unsafe_allow_html=True)
        c_kpi3.markdown(mc(f"{df_cur.isna().sum().sum():,}", "Missing Cells"), unsafe_allow_html=True)
        mem_mb = round(df_cur.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        c_kpi4.markdown(mc(f"{mem_mb} MB", "Memory Size"), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.75rem'>", unsafe_allow_html=True)
        st.dataframe(df_cur.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Diagnostic Audit Overview
        if st.session_state.audit_result is not None:
            st.markdown("### Dataset Health & ML Readiness Overview")
            render_audit_summary_cards(st.session_state.audit_result)

        render_step_navigation(
            next_step=2,
            next_label="Proceed to Data Profiling & Health ▶",
            next_key="btn_next_step1"
        )

    footer()


# STEP 2 — PROFILING

elif st.session_state.step == 2:
    if st.session_state.df_clean is None:
        render_empty_state(
            title="No Dataset Loaded",
            description="Please upload a dataset or select a benchmark dataset in Step 1 to explore data profiling and quality metrics.",
            cta_label="Go to Step 1 (Data Ingestion)",
            cta_step=1
        )
        st.stop()

    df = st.session_state.df_clean
    sh("", "Data Profiling & Quality Audit", "Understand your dataset, quality risks, and ML readiness")
    
    # Run or refresh diagnostic audit on the current dataset state
    audit = run_data_audit(df)
    st.session_state.audit_result = audit

    t0,t1,t2,t3,t4 = st.tabs(["ML Readiness Audit","Column Schema","Numeric Statistics","Categorical Statistics","Missing Patterns"])
    with t0:
        render_audit_full_dashboard(audit, key_suffix="profiling")
    with t1:
        st.dataframe(get_column_summary(df), use_container_width=True, height=400)
    with t2:
        ns = get_numeric_stats(df)
        if not ns.empty:
            st.dataframe(ns, use_container_width=True, height=350)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                sel = st.selectbox("Column for distribution", num_cols, key="dist_col")
                fig = px.histogram(df, x=sel, nbins=40, color_discrete_sequence=["#e78a53"])
                fig.update_layout(**PT, title=f"Distribution — {sel}")
                st.plotly_chart(fig, use_container_width=True)
        else: st.info("No numeric columns.")
    with t3:
        cs = get_categorical_stats(df)
        if not cs.empty: st.dataframe(cs, use_container_width=True, height=300)
        else: st.info("No categorical columns.")
    with t4:
        mp = (df.isna().sum()/len(df)*100).reset_index()
        mp.columns = ["Column","Missing %"]; mp = mp[mp["Missing %"]>0]
        if not mp.empty:
            fig = px.bar(mp, x="Missing %", y="Column", orientation="h",
                         color="Missing %", color_continuous_scale=["#5f8787","#ef4444"])
            fig.update_layout(**PT, title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)
        else: st.success("No missing values detected in dataset.")
    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) > 1:
        st.markdown("#### Feature Correlation Matrix")
        corr = num_df.corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0,"#5f8787"],[.5,"#181719"],[1,"#e78a53"]],
            text=np.round(corr.values,2), texttemplate="%{text}"))
        fig.update_layout(**PT, title="Correlation Matrix"); st.plotly_chart(fig, use_container_width=True)
    render_step_navigation(
        back_step=1,
        back_label="◀ Back",
        next_step=3,
        next_label="Next → Missing Values ▶"
    )
    footer()


# STEP 3 — MISSING VALUES

elif st.session_state.step == 3:
    if st.session_state.df_clean is None:
        render_empty_state(
            title="No Dataset Loaded",
            description="Please upload a dataset or select a benchmark dataset in Step 1 to configure missing value imputation.",
            cta_label="Go to Step 1 (Data Ingestion)",
            cta_step=1
        )
        st.stop()

    df = st.session_state.df_clean
    sh("", "Handle Missing Values", "Choose strategy per column")
    miss_df = get_missing_summary(df)
    if miss_df.empty:
        st.markdown("""<div class="ai-card">
        <span class="badge badge-success">CLEAN</span>
        <strong> No missing values detected!</strong> Dataset is already complete.</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"**{len(miss_df)} column(s) have missing values:**")
        st.dataframe(miss_df, use_container_width=True)
        suggested = suggest_missing_strategy(df)
        st.markdown("#### Configure Fill Strategy")
        st.markdown("""<div class="ai-card"><strong>Automated Strategy Recommendation:</strong>
        Strategies inferred based on data skewness and distribution. Customize per column as required.</div>""", unsafe_allow_html=True)
        strategies, custom_vals = {}, {}
        for _, row in miss_df.iterrows():
            col = row["column"]; dtype = row["dtype"]
            is_num = ("int" in dtype or "float" in dtype)
            sug = suggested.get(col, "mean" if is_num else "unknown")
            opts = (["mean","median","mode","ffill","bfill","zero","constant"]
                    if is_num else ["unknown","mode","ffill","bfill","empty","constant"])
            idx = opts.index(sug) if sug in opts else 0
            with st.expander(f"{col} — {row['missing_%']}% missing ({dtype})", expanded=True):
                cc1,cc2 = st.columns([2,3])
                with cc1:
                    badge = "badge-error" if row["missing_%"]>30 else "badge-warning"
                    st.markdown(f'<span class="badge {badge}">{row["missing_%"]}% missing</span>'
                                f'<span class="badge badge-primary" style="margin-left:.3rem">{dtype}</span>',
                                unsafe_allow_html=True)
                    choice = st.selectbox("Strategy", opts, index=idx, key=f"s_{col}")
                    strategies[col] = choice
                    if choice == "constant":
                        cv = st.text_input(f"Value for `{col}`", key=f"cv_{col}")
                        if cv:
                            try: custom_vals[col] = float(cv) if is_num else cv
                            except Exception: custom_vals[col] = cv
                with cc2:
                    sample = df[col].dropna().head(5).tolist()
                    st.markdown(f"**Sample:** `{sample}`")
                    st.markdown(f"Recommended: **`{sug}`**")
                    st.caption(f"Missing: {int(row['missing_count'])} of {len(df)} rows")
        if st.button("Apply Strategies & Fill", key="btn_apply_missing"):
            try:
                df_filled = fill_missing_values(df, strategies, custom_vals)
                remaining = int(df_filled.isna().sum().sum())
                st.session_state.df_clean = df_filled
                st.session_state.missing_applied = True
                st.session_state.preprocessing_log.append("Missing values filled")
                if remaining == 0: st.success("All missing values filled successfully!")
                else: st.warning(f"{remaining} values remain — check constant fields.")
                st.rerun()
            except Exception as e:
                render_structured_error("Imputation Failed", "Error encountered during missing value filling.", str(e), "Check data types or constant values.")
    render_step_navigation(
        back_step=2,
        back_label="◀ Back",
        next_step=4,
        next_label="Next → Duplicates ▶"
    )
    footer()


# STEP 4 — DUPLICATES

elif st.session_state.step == 4:
    if st.session_state.df_clean is None:
        render_empty_state(
            title="No Dataset Loaded",
            description="Please upload a dataset or select a benchmark dataset in Step 1 to manage duplicates.",
            cta_label="Go to Step 1 (Data Ingestion)",
            cta_step=1
        )
        st.stop()

    df = st.session_state.df_clean
    sh("", "Duplicate Record Management", "Detect and safely deduplicate identical rows or key subsets")
    exact_dup = int(df.duplicated(keep="first").sum())
    pct = round(exact_dup/len(df)*100, 2) if len(df)>0 else 0
    c1,c2,c3 = st.columns(3)
    c1.markdown(mc(f"{exact_dup:,}", "Duplicate Rows", "warning" if exact_dup > 0 else "success"), unsafe_allow_html=True)
    c2.markdown(mc(f"{pct}%", "Duplicate Percentage"), unsafe_allow_html=True)
    c3.markdown(mc(f"{len(df)-exact_dup:,}", "Cleaned Row Count", "primary"), unsafe_allow_html=True)
    if exact_dup == 0:
        st.markdown("""<div class="ai-card">
        <span class="badge badge-success">CLEAN</span>
        <strong> No duplicate records detected!</strong> Every row in the dataset is unique.</div>""", unsafe_allow_html=True)
    else:
        with st.expander("Preview Duplicate Rows", expanded=False):
            st.dataframe(df[df.duplicated(keep=False)].head(50), use_container_width=True)
        cc1,cc2 = st.columns(2)
        with cc1:
            keep_opt = st.selectbox("Which copy to keep?",
                ["first — keep first occurrence","last  — keep last occurrence","none  — remove ALL copies"])
            keep_val = (False if "none" in keep_opt else "first" if "first" in keep_opt else "last")
        with cc2:
            use_subset = st.checkbox("Check only specific columns", value=False)
            subset = None
            if use_subset:
                sel_cols = st.multiselect("Columns to check", df.columns.tolist())
                subset = sel_cols if sel_cols else None
                if subset:
                    sub_dup = int(df.duplicated(subset=subset, keep="first").sum())
                    st.info(f"With selected columns: **{sub_dup}** duplicates")
        st.markdown("""<div class="ai-card"><strong>Deduplication Guidance:</strong>
        Full-row comparison is safest. Use column subset only for key-based deduplication.</div>""",
        unsafe_allow_html=True)
        if st.button("Remove Duplicate Rows", key="btn_remove_dup"):
            before = len(df)
            df_clean = df.drop_duplicates(subset=subset, keep=keep_val)
            after = len(df_clean)
            st.session_state.df_clean = df_clean
            st.session_state.dup_removed = True
            st.session_state.preprocessing_log.append(f"Removed {before-after} duplicate rows")
            st.success(f"Removed **{before-after}** duplicate rows. {before:,} → {after:,}")
            time.sleep(0.3); st.rerun()
    def _on_next_to_prep():
        st.session_state.df_cleaned_only = st.session_state.df_clean.copy()

    render_step_navigation(
        back_step=3,
        back_label="◀ Back",
        next_step=5,
        next_label="Next → Preprocessing ▶",
        on_next_click=_on_next_to_prep
    )
    footer()


# STEP 5 — PREPROCESSING

elif st.session_state.step == 5:
    if st.session_state.df_clean is None:
        render_empty_state(
            title="No Dataset Loaded",
            description="Please upload a dataset or select a benchmark dataset in Step 1 to configure feature preprocessing.",
            cta_label="Go to Step 1 (Data Ingestion)",
            cta_step=1
        )
        st.stop()

    df = st.session_state.df_clean
    sh("", "Preprocessing", "Encode, scale, normalize & engineer your features")

    #  Smart AI Guide ─
    st.markdown("### AI Preprocessing Recommendations")
    ai_rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_total  = len(df)
        is_num   = pd.api.types.is_numeric_dtype(df[col])
        is_cat   = dtype in ("object","category") or not is_num

        if is_num:
            col_data = df[col].dropna()
            try:
                rng   = float(col_data.max() - col_data.min())
                skew  = float(col_data.skew())
                std   = float(col_data.std())
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

        ai_rows.append({"Column": col, "Type": dtype, "Unique Values": n_unique,
                        "Recommended Action": action, "Reason": reason})

    ai_guide_df = pd.DataFrame(ai_rows)

    def _style_action(val):
        if "Drop" in val:   return "background:rgba(255,107,107,.15);color:#ff6b6b"
        if "Scaler" in val: return "background:rgba(0,212,170,.15);color:#00d4aa"
        if "Encoding" in val: return "background:rgba(124,92,252,.15);color:#a78bfa"
        return "background:rgba(255,179,71,.1);color:#ffb347"

    styled = ai_guide_df.style.map(_style_action, subset=["Recommended Action"])
    st.dataframe(styled, use_container_width=True, height=min(35*len(ai_guide_df)+40, 380))
    st.markdown("""<div class="ai-card" style="font-size:.82rem">
    <strong>Preprocessing is optional.</strong> The recommendations above are suggestions based on your data distribution.
    Use the tabs below to apply transformations as needed, then click <strong>Next</strong> to proceed.
    The original cleaned dataset is preserved separately.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    all_cols   = df.columns.tolist()
    num_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols   = df.select_dtypes(include=["object","category"]).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs(["Encoding", "Scaling / Normalization", "Drop Columns", "Feature Info"])

    with tab1:
        st.markdown("#### Encode Categorical Columns")
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            enc_method = st.selectbox("Encoding method",
                ["Label Encoding","One-Hot Encoding","Ordinal Encoding"], key="enc_method")
            enc_cols = st.multiselect("Select columns to encode", cat_cols, default=cat_cols, key="enc_cols")
            if st.button("Apply Encoding"):
                try:
                    df_enc = st.session_state.df_clean.copy()
                    log_msg = []
                    if enc_method == "Label Encoding":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        for col in enc_cols:
                            if col in df_enc.columns:
                                df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                        log_msg.append(f"Label encoded: {enc_cols}")
                    elif enc_method == "One-Hot Encoding":
                        df_enc = pd.get_dummies(df_enc, columns=enc_cols, drop_first=False)
                        log_msg.append(f"One-hot encoded: {enc_cols}")
                    elif enc_method == "Ordinal Encoding":
                        from sklearn.preprocessing import OrdinalEncoder
                        oe = OrdinalEncoder()
                        df_enc[enc_cols] = oe.fit_transform(df_enc[enc_cols].astype(str))
                        log_msg.append(f"Ordinal encoded: {enc_cols}")
                    st.session_state.df_clean = df_enc
                    st.session_state.preprocessing_log.extend(log_msg)
                    st.success(f"{enc_method} applied to {len(enc_cols)} columns.")
                    st.dataframe(df_enc.head(5), use_container_width=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Encoding error: {e}")

    with tab2:
        st.markdown("#### Scale / Normalize Numeric Columns")
        if not num_cols:
            st.info("No numeric columns found.")
        else:
            scale_method = st.selectbox("Scaling method",
                ["StandardScaler (mean=0, std=1)",
                 "MinMaxScaler (0 to 1)",
                 "RobustScaler (outlier-resistant)",
                 "Normalizer (unit norm per row)"], key="scale_method")
            scale_cols = st.multiselect("Columns to scale", num_cols, default=num_cols[:3] if len(num_cols)>3 else num_cols, key="scale_cols")
            if st.button("Apply Scaling"):
                try:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
                    df_sc = st.session_state.df_clean.copy()
                    if scale_cols:
                        method_map = {
                            "StandardScaler (mean=0, std=1)": StandardScaler(),
                            "MinMaxScaler (0 to 1)": MinMaxScaler(),
                            "RobustScaler (outlier-resistant)": RobustScaler(),
                            "Normalizer (unit norm per row)": Normalizer(),
                        }
                        scaler = method_map[scale_method]
                        df_sc[scale_cols] = scaler.fit_transform(df_sc[scale_cols].fillna(0))
                        st.session_state.df_clean = df_sc
                        st.session_state.preprocessing_log.append(f"{scale_method} on {scale_cols}")
                        st.success(f"Scaling applied to {len(scale_cols)} columns.")
                        st.dataframe(df_sc[scale_cols].head(5), use_container_width=True)
                except Exception as e:
                    st.error(f"Scaling error: {e}")

    with tab3:
        st.markdown("#### Drop Columns")
        drop_cols = st.multiselect("Select columns to drop", st.session_state.df_clean.columns.tolist(), key="drop_cols")
        if drop_cols:
            st.warning(f"Will drop: **{drop_cols}**")
            if st.button("Drop Selected Columns"):
                df_drop = st.session_state.df_clean.drop(columns=drop_cols)
                st.session_state.df_clean = df_drop
                st.session_state.preprocessing_log.append(f"Dropped columns: {drop_cols}")
                st.success(f"Dropped {len(drop_cols)} column(s). Remaining: {len(df_drop.columns)}.")
                st.rerun()

    with tab4:
        st.markdown("#### Preprocessed Dataset Summary")
        df_now = st.session_state.df_clean
        info = get_basic_info(df_now)
        c1,c2,c3 = st.columns(3)
        c1.markdown(mc(f"{info['rows']:,}", "Rows"), unsafe_allow_html=True)
        c2.markdown(mc(f"{info['columns']:,}", "Columns"), unsafe_allow_html=True)
        c3.markdown(mc(f"{int(df_now.isna().sum().sum()):,}", "Missing"), unsafe_allow_html=True)
        st.dataframe(get_column_summary(df_now), use_container_width=True, height=300)
        if st.session_state.preprocessing_log:
            st.markdown("#### Preprocessing Action Log")
            for i,log in enumerate(st.session_state.preprocessing_log):
                st.markdown(f"`{i+1}.` {log}")

    render_step_navigation(
        back_step=4,
        back_label="◀ Back",
        next_step=6,
        next_label="Next → Export Clean Data ▶"
    )
    footer()


# STEP 6 — EXPORT CLEAN DATA (dual download + training selector)

elif st.session_state.step == 6:
    if st.session_state.df_clean is None:
        render_empty_state(
            title="No Dataset Loaded",
            description="Please upload a dataset or select a benchmark dataset in Step 1 before exporting data.",
            cta_label="Go to Step 1 (Data Ingestion)",
            cta_step=1
        )
        st.stop()

    sh("", "Export Clean Dataset", "Download cleaned & preprocessed data — choose which to use for training")

    df_preprocessed = st.session_state.df_clean
    df_cleaned      = st.session_state.df_cleaned_only if st.session_state.df_cleaned_only is not None else df_preprocessed

    EXT  = {"csv":"csv","excel":"xlsx","json":"json","xml":"xml","yaml":"yaml","sql":"db"}
    MIME = {"csv":"text/csv","excel":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "json":"application/json","xml":"application/xml","yaml":"text/yaml","sql":"application/octet-stream"}
    FMT_LABELS = {"csv":"CSV","excel":"Excel (.xlsx)","json":"JSON",
                  "xml":"XML","yaml":"YAML","sql":"SQLite DB"}

    def _make_download_bytes(df, fmt):
        try:
            if fmt == "sql":
                import sqlite3, tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db"); tmp.close()
                sdf = df.copy()
                sdf.columns = [str(c).replace(" ","_").replace(".","_").replace("-","_") for c in sdf.columns]
                conn = sqlite3.connect(tmp.name)
                sdf.to_sql("data", conn, if_exists="replace", index=False)
                conn.commit(); conn.close()
                with open(tmp.name,"rb") as f: raw = f.read()
                os.remove(tmp.name)
                return raw
            else:
                raw, _ = export_data(df, fmt)
                return raw.encode("utf-8") if isinstance(raw, str) else raw
        except Exception as ex:
            st.error(f"Export error: {ex}"); return None

    col_a, col_b = st.columns(2)

    # LEFT: Cleaned-Only
    with col_a:
        st.markdown("""<div class="glass-card" style="margin-bottom:1rem">
        <div style="font-size:0.75rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1px">DATASET STAGE 1</div>
        <div style="font-weight:700;color:var(--text);font-size:1.1rem;margin-top:0.2rem">Cleaned Data</div>
        <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:0.3rem">After missing & duplicate resolution — before encoding & scaling</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"**Dimensions:** `{len(df_cleaned):,}` rows &times; `{len(df_cleaned.columns)}` columns")
        st.dataframe(df_cleaned.head(5), use_container_width=True)
        fmt_a = st.selectbox("Format", list(EXT.keys()), format_func=lambda x: FMT_LABELS.get(x,x), key="fmt_cleaned")
        if st.button("Export Cleaned Data", use_container_width=True, key="dl_cleaned"):
            with st.spinner("Preparing export package..."):
                raw = _make_download_bytes(df_cleaned, fmt_a)
                if raw:
                    st.download_button(f"Download cleaned_data.{EXT[fmt_a]}", data=raw,
                        file_name=f"cleaned_data.{EXT[fmt_a]}", mime=MIME[fmt_a], key="dlb_cleaned")
                    st.success("Cleaned dataset ready for download.")

    # RIGHT: Preprocessed
    with col_b:
        st.markdown("""<div class="glass-card" style="margin-bottom:1rem">
        <div style="font-size:0.75rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1px">DATASET STAGE 2</div>
        <div style="font-weight:700;color:var(--accent);font-size:1.1rem;margin-top:0.2rem">Preprocessed Data</div>
        <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:0.3rem">After encoding, scaling & all active transformation steps</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"**Dimensions:** `{len(df_preprocessed):,}` rows &times; `{len(df_preprocessed.columns)}` columns")
        st.dataframe(df_preprocessed.head(5), use_container_width=True)
        fmt_b = st.selectbox("Format", list(EXT.keys()), format_func=lambda x: FMT_LABELS.get(x,x), key="fmt_preprocessed")
        if st.button("Export Preprocessed Data", use_container_width=True, key="dl_preprocessed"):
            with st.spinner("Preparing export package..."):
                raw = _make_download_bytes(df_preprocessed, fmt_b)
                if raw:
                    st.download_button(f"Download preprocessed_data.{EXT[fmt_b]}", data=raw,
                        file_name=f"preprocessed_data.{EXT[fmt_b]}", mime=MIME[fmt_b], key="dlb_preprocessed")
                    st.success("Preprocessed dataset ready for download.")

    # Training Dataset Selector
    st.markdown("---")
    st.markdown("### Training Dataset Configuration")
    st.markdown("""<div class="ai-card">
    <strong>Pipeline Recommendation:</strong> Use <strong>Preprocessed Data</strong> if you applied encoding/scaling —
    models require numeric inputs. Use <strong>Cleaned Data</strong> if you prefer the raw cleaned format
    (the AutoML pipeline will handle categorical encodings automatically).
    </div>""", unsafe_allow_html=True)
    train_choice = st.radio(
        "Which dataset should Step 7 (Model Training) use?",
        ["preprocessed", "cleaned"],
        index=0 if st.session_state.training_dataset == "preprocessed" else 1,
        format_func=lambda x: "Preprocessed Data (Recommended)" if x=="preprocessed" else "Cleaned Data (Raw Features)",
        horizontal=True, key="train_ds_radio"
    )
    st.session_state.training_dataset = train_choice

    render_step_navigation(
        back_step=5,
        back_label="◀ Back",
        next_step=7,
        next_label="Proceed to Model Training ▶"
    )
    footer()


# STEP 7 — MODEL TRAINING + RESULTS

elif st.session_state.step == 7:
    if st.session_state.df_clean is None:
        render_empty_state(
            title="No Dataset Available for Training",
            description="Please upload a dataset or select a benchmark dataset in Step 1 before configuring model training.",
            cta_label="Go to Step 1 (Data Ingestion)",
            cta_step=1
        )
        st.stop()

    # Use dataset chosen in Step 6
    _use_prep = st.session_state.training_dataset == "preprocessed"
    df = st.session_state.df_clean if _use_prep else (
        st.session_state.df_cleaned_only if st.session_state.df_cleaned_only is not None
        else st.session_state.df_clean)
    sh("", "Model Training & Experiment Workspace", "Configure, train, and evaluate leakage-safe machine learning models")

    # Dataset badge
    _ds_label = "Preprocessed Dataset" if _use_prep else "Cleaned Dataset"
    st.markdown(f"""<div class="glass-card" style="font-size:.84rem;padding:.7rem 1.1rem;margin-bottom:1rem">
    <strong>Training on:</strong> <span class="badge badge-success">{_ds_label}</span>
    &nbsp;&bull;&nbsp; {len(df):,} rows &times; {len(df.columns)} columns
    &nbsp;<span style="color:var(--muted);font-size:.75rem">(configured in Step 6)</span>
    </div>""", unsafe_allow_html=True)

    all_cols = df.columns.tolist()

    # AI Analysis
    _default_target = st.session_state.target_col if st.session_state.target_col in all_cols else all_cols[-1]
    _ai_target = _default_target

    if st.session_state.ai_report is None or st.session_state.target_col != _ai_target:
        with st.spinner("Analyzing dataset characteristics..."):
            report = generate_ai_report(df, _ai_target)
        st.session_state.ai_report = report
        st.session_state.target_col = _ai_target
    report = st.session_state.ai_report

    with st.expander("Automated Dataset Intelligence & Recommendations", expanded=False):
        cc = st.columns(4)
        cc[0].markdown(mc(f"{report['dataset']['rows']:,}", "Rows"), unsafe_allow_html=True)
        cc[1].markdown(mc(f"{report['dataset']['columns']:,}", "Columns"), unsafe_allow_html=True)
        cc[2].markdown(mc(f"{report['dataset']['missing_cells']:,}", "Missing"), unsafe_allow_html=True)
        cc[3].markdown(mc(f"{report['dataset']['duplicate_rows']:,}", "Duplicates"), unsafe_allow_html=True)
        pt_ai      = report.get("problem_type") or detect_problem_type(df, _ai_target)
        rec_models = report.get("recommended_models", [])
        imp_feat   = report.get("important_features", [])
        clust_rec  = report.get("clustering", {})
        miss_strat = report.get("missing_strategy", {})
        ptc = "badge-primary" if pt_ai=="classification" else "badge-success"
        st.markdown(f"""<div class="ai-card" style="margin-top:1rem">
        <strong>Automated Architecture Recommendations</strong><br><br>
        <strong>Problem Type:</strong> <span class="badge {ptc}">{pt_ai.upper() if pt_ai else "N/A"}</span><br><br>
        <strong>Recommended Estimators:</strong> {", ".join(rec_models[:5]) or "N/A"}<br><br>
        <strong>Influential Correlated Features:</strong> {", ".join([str(f) for f in imp_feat[:5]]) or "N/A"}<br><br>
        <strong>Cluster Structure:</strong> {clust_rec.get("recommended","N/A")} — {clust_rec.get("reason","")}<br><br>
        <strong>Imputation Strategy:</strong> {", ".join([f"{k}→{v}" for k,v in list(miss_strat.items())[:4]]) or "Complete data (no imputation needed)"}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Target Column + Task Type
    target_col = st.selectbox("Target Column (Prediction Objective)", all_cols,
        index=all_cols.index(_default_target) if _default_target in all_cols else len(all_cols)-1)

    # Refresh AI if target changed
    if target_col != st.session_state.target_col:
        with st.spinner("Re-evaluating target..."):
            report = generate_ai_report(df, target_col)
        st.session_state.ai_report = report
        st.session_state.target_col = target_col
        report = st.session_state.ai_report

    auto_type = detect_problem_type(df, target_col)

    task_opts = ["classification", "regression"]
    task_type = st.selectbox("Task Type", task_opts,
        index=task_opts.index(auto_type) if auto_type in task_opts else 0)

    if auto_type != task_type:
        st.markdown(f"""<div class="warn-card">
        <strong>Advisory:</strong> Pipeline inferred <span class="badge badge-success">{auto_type}</span>
        for this target column, but you configured <span class="badge badge-primary">{task_type}</span>.
        Ensure your chosen target matches the desired task objective.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='font-size:0.8rem;color:var(--muted)'>Inferred Task:</span> <span class='badge badge-primary'>{auto_type.upper()}</span>", unsafe_allow_html=True)

    # Pre-Flight Target & Quality Audit Check
    target_audit = run_data_audit(df, target_col=target_col)
    st.session_state.audit_target_result = target_audit
    render_target_preflight_check(target_audit)

    # Validation Setup & Configurable Optimization Metric
    st.markdown("#### Validation & Optimization Setup")
    avail_metrics = get_available_metrics(task_type)
    default_metric = "F1 Score" if task_type == "classification" else "R2 Score"
    default_idx = avail_metrics.index(default_metric) if default_metric in avail_metrics else 0

    cfg1, cfg2, cfg3 = st.columns([2, 1, 1])
    with cfg1:
        opt_metric = st.selectbox(
            "Optimization Metric (Leaderboard Ranker)",
            avail_metrics,
            index=default_idx,
            key=f"opt_metric_{task_type}"
        )
        higher_better = is_higher_better(opt_metric)
        direction_badge = "badge-success" if higher_better else "badge-primary"
        direction_text = "Higher is better" if higher_better else "Lower is better"
        desc_text = METRIC_CONFIGS.get(opt_metric, {}).get("description", "")
        st.markdown(f"<span class='badge {direction_badge}'>{direction_text}</span> &nbsp; <span style='font-size:0.75rem;color:var(--muted)'>{desc_text}</span>", unsafe_allow_html=True)

    with cfg2:
        cv_folds_sel = st.slider("Cross-Validation Folds", min_value=2, max_value=10, value=5, key=f"cv_folds_{task_type}")

    with cfg3:
        test_size_sel = st.slider("Holdout Test %", min_value=10, max_value=40, value=20, step=5, key=f"test_size_{task_type}")

    # Two-Stage AutoML & Hyperparameter Optimization Controls
    st.markdown("#### Two-Stage AutoML & Hyperparameter Optimization")
    tune_c1, tune_c2, tune_c3 = st.columns([2, 1, 1])
    with tune_c1:
        enable_tuning_sel = st.checkbox("Enable Stage B Hyperparameter Optimization", value=False, key=f"enable_tuning_{task_type}")
    with tune_c2:
        tune_top_k_sel = st.selectbox("Top K Models to Tune", [1, 2, 3], index=1, key=f"tune_top_k_{task_type}")
    with tune_c3:
        tune_n_iter_sel = st.slider("Search Iterations", min_value=5, max_value=20, value=8, step=1, key=f"tune_n_iter_{task_type}")

    st.markdown("#### Select Estimator Candidates")
    if task_type == "classification":
        MODEL_KEYS = [
            "logistic_regression", "ridge_classifier", "decision_tree",
            "random_forest", "extra_trees", "gradient_boosting",
            "hist_gradient_boosting", "svc", "knn", "naive_bayes"
        ]
        MODEL_DISP = [
            "Logistic Regression", "Ridge Classifier", "Decision Tree",
            "Random Forest", "Extra Trees", "Gradient Boosting",
            "HistGradientBoosting", "Support Vector (SVC)", "K-Nearest Neighbors", "Gaussian Naive Bayes"
        ]
    else:
        MODEL_KEYS = [
            "linear_regression", "ridge", "lasso", "elastic_net",
            "decision_tree", "random_forest", "extra_trees",
            "gradient_boosting", "hist_gradient_boosting", "svr"
        ]
        MODEL_DISP = [
            "Linear Regression", "Ridge", "Lasso", "ElasticNet",
            "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting",
            "HistGradientBoosting", "SVR"
        ]

    sel_keys = []
    cols3 = st.columns(3)
    for i,(disp,key) in enumerate(zip(MODEL_DISP,MODEL_KEYS)):
        with cols3[i%3]:
            if st.checkbox(disp, value=True, key=f"m_{key}_{task_type}"):
                sel_keys.append(key)

    if st.button("Train Models", use_container_width=True):
        if not sel_keys:
            st.error("Select at least one candidate estimator.")
            st.stop()
        try:
            is_imbalanced = False
            if st.session_state.audit_target_result and st.session_state.audit_target_result.target_analysis:
                is_imbalanced = st.session_state.audit_target_result.target_analysis.is_severely_imbalanced

            X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
                df,
                target_col=target_col,
                test_size=test_size_sel / 100.0,
                random_state=42,
                problem_type=task_type
            )

            prog = st.progress(0, text="1/5 • Preparing pipeline and stratified split...")
            def progress_cb(pct, txt):
                prog.progress(pct, text=txt)

            automl_result = train_and_evaluate_models(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                problem_type=task_type,
                selected_models=sel_keys,
                optimization_metric=opt_metric,
                cv_folds=cv_folds_sel,
                is_imbalanced=is_imbalanced,
                enable_tuning=enable_tuning_sel,
                tune_top_k=tune_top_k_sel,
                tune_n_iter=tune_n_iter_sel,
                progress_callback=progress_cb
            )
            prog.empty()

            st.session_state.results_df = automl_result.results_df
            st.session_state.trained_models = automl_result.trained_models
            st.session_state.best_model_name = automl_result.best_model_name
            st.session_state.selection_reason = automl_result.selection_reason
            st.session_state.provenance = automl_result.provenance
            st.session_state.problem_type = task_type
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.feature_schema = getattr(automl_result, "feature_schema", extract_feature_schema(X_train))
            st.session_state.optimization_metric = opt_metric
            st.session_state.cv_folds = cv_folds_sel
            st.session_state.test_size_pct = test_size_sel
            st.session_state.tuning_enabled = enable_tuning_sel
            st.session_state.shap_cache = {}
            st.success(f"Training completed successfully! Optimal Model: **{automl_result.best_model_name}** ({opt_metric}: {automl_result.results_df.iloc[0].get(opt_metric, 'N/A') if not automl_result.results_df.empty else 'N/A'})")
            st.rerun()
        except Exception as e:
            render_structured_error(
                title="Model Training Encountered an Issue",
                what_happened="The training pipeline could not complete evaluation with the selected configuration.",
                why=str(e),
                what_to_do="Review target column encoding, check for single-class targets or high-cardinality features, or reduce cross-validation folds.",
                debug_trace=traceback.format_exc()
            )

    # Results displayed after training
    if (st.session_state.results_df is not None and
            st.session_state.problem_type in ["classification","regression"]):
        results_df     = st.session_state.results_df
        trained_models = st.session_state.trained_models
        best_name      = st.session_state.best_model_name
        problem_type   = st.session_state.problem_type
        best_obj       = trained_models.get(best_name)
        opt_metric_cur = st.session_state.get("optimization_metric") or ("F1 Score" if problem_type=="classification" else "R2 Score")
        exp_rec        = recommend_model_export(best_obj) if best_obj else {}
        sel_reason_cur = st.session_state.get("selection_reason", "")
        prov_meta      = st.session_state.get("provenance", {})

        st.markdown("---")
        st.markdown(f"""<div class="glass-panel" style="border-left:4px solid var(--primary);margin-bottom:1.25rem">
            <div style="font-size:0.7rem;font-weight:700;color:var(--muted-foreground);text-transform:uppercase;letter-spacing:1.5px">MODEL SELECTION CRITERION</div>
            <div style="font-size:1.25rem;font-weight:800;color:var(--foreground);margin-top:0.25rem;display:flex;align-items:center;gap:0.6rem">
                Selected Model: <span style="color:var(--primary)">{best_name}</span>
                <span class="badge badge-success">OPTIMAL</span>
            </div>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.4rem;line-height:1.5">
                <strong>Configured Metric:</strong> <code>{opt_metric_cur}</code> &nbsp;&bull;&nbsp;
                Model selected based on the configured optimization metric ({'higher is better' if is_higher_better(opt_metric_cur) else 'lower is better'}).
                Encapsulates full preprocessing pipeline fitted strictly on training data.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Model Leaderboard")
        st.markdown(f"""<div style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:0.6rem">
        Evaluated across <strong>{st.session_state.get('cv_folds', 5)} cross-validation folds</strong> and tested on an independent <strong>{st.session_state.get('test_size_pct', 20)}% holdout test set</strong>.
        Sorted deterministically by <strong>{opt_metric_cur}</strong>.
        </div>""", unsafe_allow_html=True)

        def highlight_best_row(row):
            if row.get("Model") == best_name:
                return ["background-color: rgba(231, 138, 83, 0.18); font-weight: 600"] * len(row)
            return [""] * len(row)

        styled_df = results_df.style.apply(highlight_best_row, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        if prov_meta:
            with st.expander("Reproducibility & Run Provenance Details", expanded=False):
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.markdown(f"**Random Seed:** `{prov_meta.get('random_state', 42)}`")
                    st.markdown(f"**CV Folds:** `{prov_meta.get('cv_folds', 5)}`")
                    st.markdown(f"**Tuning Enabled:** `{'Yes' if prov_meta.get('tuning_enabled') else 'No'}`")
                with p2:
                    dim = prov_meta.get("dataset_dimensions", {})
                    st.markdown(f"**Train Rows:** `{dim.get('train_rows', 'N/A')}`")
                    st.markdown(f"**Test Rows:** `{dim.get('test_rows', 'N/A')}`")
                    st.markdown(f"**Features:** `{dim.get('feature_count', 'N/A')}`")
                with p3:
                    st.markdown(f"**Timestamp (UTC):** `{prov_meta.get('timestamp', 'N/A')[:19]}`")
                    st.markdown(f"**Optimization Metric:** `{prov_meta.get('optimization_metric', 'N/A')}`")
                
                b_params = prov_meta.get("best_hyperparameters", {})
                if b_params and isinstance(b_params, dict) and len(b_params) > 0:
                    st.markdown(f"**Best Model Hyperparameters ({best_name}):**")
                    st.json(b_params)

        metric_col = opt_metric_cur if opt_metric_cur in results_df.columns else ("F1 Score" if problem_type=="classification" else "R2 Score")
        if metric_col in results_df.columns:
            fig = px.bar(results_df, x="Model", y=metric_col,
                         color=metric_col, color_continuous_scale=["#5f8787", "#e78a53"], text=metric_col)
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(**PT, title=f"{metric_col} Comparison across Models")
            st.plotly_chart(fig, use_container_width=True)

        if len(num_res) >= 3:
            fig = go.Figure()
            for _,row in results_df.iterrows():
                vals = [row[m] for m in num_res]
                fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=num_res+[num_res[0]],
                    name=row["Model"], fill='toself', opacity=0.4))
            fig.update_layout(**PT, title="Multivariate Leaderboard Comparison",
                polar=dict(radialaxis=dict(visible=True, color="rgba(188, 214, 217, 0.25)"), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Export Trained Pipeline (Production Format)", expanded=True):
            if best_obj:
                fc1, fc2 = st.columns([1, 2])
                with fc1:
                    fmt_opt = st.selectbox("Format", ["joblib", "pickle", "onnx"], key="export_fmt_step7")
                with fc2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    X_s = st.session_state.X_test.iloc[:5] if st.session_state.X_test is not None else None
                    data, mime = export_model(best_obj, fmt_opt, X_s)
                    if data:
                        ext = {"pickle": "pkl", "joblib": "joblib", "onnx": "onnx"}.get(fmt_opt, "bin")
                        st.download_button(
                            label=f"Download {best_name} ({ext.upper()})",
                            data=data,
                            file_name=f"{best_name.replace(' ', '_')}.{ext}",
                            mime=mime or "application/octet-stream",
                            key="btn_dl_step7"
                        )
                    else:
                        st.warning(f"Export unavailable: {mime}")

        # Model Explanation Section
        st.markdown("---")
        render_model_explanation_ui(
            pipeline_or_model=best_obj,
            model_name=best_name,
            problem_type=problem_type,
            X_sample=st.session_state.get("X_test"),
            y_sample=st.session_state.get("y_test"),
            X_background=st.session_state.get("X_train")
        )

        # Prediction Playground Section
        st.markdown("---")
        render_prediction_playground_ui(
            trained_models=trained_models,
            default_model_name=best_name,
            problem_type=problem_type,
            feature_schema=st.session_state.get("feature_schema"),
            X_background=st.session_state.get("X_train"),
            target_col=st.session_state.get("target_col")
        )

    st.markdown("---")
    render_step_navigation(
        back_step=6,
        back_label="◀ Back",
        next_step=8,
        next_label="Next → Cluster Visualizer ▶",
        extra_action=("Open Dedicated Playground ▶", 10, "btn_step7_playground")
    )
    footer()


# STEP 8 — CLUSTER VISUALIZER (redesigned — no 3D)

elif st.session_state.step == 8:
    if st.session_state.df_clean is None:
        render_empty_state(
            title="No Dataset Available for Clustering",
            description="Please upload a dataset or load a benchmark dataset in Step 1 before analyzing clusters and feature projections.",
            cta_label="Go to Step 1 (Data Ingestion)",
            cta_step=1
        )
        st.stop()

    sh("", "Cluster Visualizer", "Clustering analysis + interactive data graph builder")

    df_c = st.session_state.df_clean
    _, scaled = prepare_clustering_data(df_c)

    if scaled is None:
        st.warning("Not enough numeric data for clustering. Need ≥2 numeric columns & ≥5 rows.")
    else:
        #  SECTION A: Clustering ─
        st.markdown("### Clustering Analysis")
        clust_rec = recommend_clustering(df_c)

        # Run all 3 algorithms
        if st.session_state.cluster_results is None:
            with st.spinner("Running KMeans, DBSCAN & Agglomerative..."):
                st.session_state.cluster_results = run_all_clustering(scaled)
        all_clust = st.session_state.cluster_results

        reduced = reduce_to_2d(scaled)
        best_c, best_cs = get_best_clustering(all_clust)

        # Build display df
        num_orig = df_c.select_dtypes(include=[np.number]).dropna()
        n = min(len(reduced), len(num_orig)) if reduced is not None else len(num_orig)
        ddf = num_orig.iloc[:n].reset_index(drop=True).copy()
        if reduced is not None:
            ddf["PC1"] = reduced[:n,0]; ddf["PC2"] = reduced[:n,1]
        num_c = [c for c in ddf.columns if c not in ["PC1","PC2"]]

        # Clustering selector + AI guide
        algo_col, info_col = st.columns([1,2])
        with algo_col:
            st.markdown("#### Choose Algorithm")
            algo_sel = st.selectbox("Clustering Algorithm",
                ["KMeans","DBSCAN","Agglomerative"], key="clust_algo_main")
        with info_col:
            algo_descs = {
                "KMeans": ("Best for large datasets with spherical clusters. Needs K (number of clusters) set manually.", "badge-green"),
                "DBSCAN": ("Best for irregular shapes & noise detection. Does not need K. Slow on large data.", "badge-purple"),
                "Agglomerative": ("Best for small datasets & hierarchical structure. Memory-intensive on large data.", "badge-yellow"),
            }
            desc, badge_cls = algo_descs[algo_sel]
            best_badge = "badge-green" if algo_sel == best_c else "badge-purple"
            rating_text = "Excellent" if best_cs > 0.5 else "Moderate" if best_cs > 0.25 else "Poor"
            st.markdown(f"""<div class="ai-card" style="margin-top:1.5rem">
            <strong>AI Cluster Guide:</strong><br><br>
            <strong>Recommended:</strong> <span class="badge badge-green">{best_c}</span>
            (Silhouette = <strong>{best_cs:.4f}</strong> — {rating_text})<br><br>
            <strong>Selected ({algo_sel}):</strong> {desc}
            </div>""", unsafe_allow_html=True)

        # Silhouette scores table
        sil_df = pd.DataFrame([{"Algorithm":k,"Silhouette Score":v["score"],"Status":"Best" if k==best_c else ""}
                                for k,v in all_clust.items()]).sort_values("Silhouette Score",ascending=False)
        st.dataframe(sil_df, use_container_width=True)

        # Display clustering scatter (PC1 vs PC2)
        if reduced is not None:
            lbl_sel = [str(all_clust[algo_sel]["labels"][i]) for i in range(n)]
            ddf_plot = ddf.copy(); ddf_plot["Cluster"] = lbl_sel
            fig = px.scatter(ddf_plot, x="PC1", y="PC2", color="Cluster",
                color_discrete_sequence=COLORS, symbol="Cluster",
                title=f"{algo_sel} — Cluster Map (PCA 2D)",
                labels={"PC1":"Principal Component 1","PC2":"Principal Component 2"})
            fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5,color='#0a0a0f')))
            fig.update_layout(**PT)
            st.plotly_chart(fig, use_container_width=True)

            # Elbow curve
            with st.expander("KMeans Elbow Curve (Optimal K finder)", expanded=False):
                Ks, inertias = find_optimal_clusters(scaled)
                valid = [(k,i) for k,i in zip(Ks,inertias) if i is not None]
                if valid:
                    kv,iv = zip(*valid)
                    fig2 = go.Figure(go.Scatter(x=list(kv), y=list(iv), mode="lines+markers",
                        marker=dict(color="#7c5cfc",size=9,symbol="diamond"),
                        line=dict(color="#00d4aa",width=2.5)))
                    fig2.update_layout(**PT, title="Elbow Curve — Optimal K",
                                      xaxis_title="K (Clusters)", yaxis_title="Inertia")
                    st.plotly_chart(fig2, use_container_width=True)

        #  SECTION B: Interactive Graph Builder 
        st.markdown("---")
        st.markdown("### Interactive Graph Builder")
        st.markdown("""<div class="ai-card" style="font-size:.85rem">
        <strong>Build any chart from your data.</strong> Select chart type, then configure X/Y axes.
        Compare multiple columns, explore distributions, and spot patterns.
        </div>""", unsafe_allow_html=True)

        all_df_cols = df_c.columns.tolist()
        num_df_cols = df_c.select_dtypes(include=[np.number]).columns.tolist()
        cat_df_cols = df_c.select_dtypes(include=["object","category"]).columns.tolist()

        CHART_TYPES = ["Scatter Plot","Bar Chart","Histogram","Box Plot","Pie Chart",
                       "Violin Plot","Line Chart","Area Chart","Density (KDE)","ECDF Plot",
                       "Regression Plot","Heatmap (Correlation)","Compare Plot"]

        gc1, gc2 = st.columns(2)
        with gc1: chart_type = st.selectbox("Chart Type", CHART_TYPES, key="gb_chart")
        with gc2:
            color_by = st.selectbox("Color By (optional)", ["None"]+all_df_cols, key="gb_color")
            color_col = None if color_by == "None" else color_by

        # Axis selectors — context aware
        def ax_select(label, options, key, exclude=None):
            opts = [c for c in options if c != exclude] if exclude else options
            if not opts: opts = options
            return st.selectbox(label, opts, key=key)

        fig = None

        if chart_type == "Scatter Plot":
            a1,a2 = st.columns(2)
            with a1: xc = ax_select("X Axis", num_df_cols or all_df_cols, "sc_x")
            with a2: yc = ax_select("Y Axis", num_df_cols or all_df_cols, "sc_y", exclude=xc)
            fig = px.scatter(df_c, x=xc, y=yc, color=color_col, color_discrete_sequence=COLORS,
                title=f"Scatter: {xc} vs {yc}", trendline="ols" if len(df_c)>5 else None)
            fig.update_traces(marker=dict(size=9, opacity=0.8))

        elif chart_type == "Bar Chart":
            a1,a2 = st.columns(2)
            with a1: xc = ax_select("X (Category)", all_df_cols, "bar_x")
            with a2: yc = ax_select("Y (Value)", num_df_cols or all_df_cols, "bar_y", exclude=xc)
            barmode = st.radio("Bar mode", ["group","stack","overlay"], horizontal=True, key="bar_mode")
            fig = px.bar(df_c, x=xc, y=yc, color=color_col or xc,
                color_discrete_sequence=COLORS, barmode=barmode, title=f"Bar: {xc} vs {yc}")

        elif chart_type == "Histogram":
            a1,a2 = st.columns(2)
            with a1: xc = ax_select("Column", num_df_cols or all_df_cols, "hist_x")
            with a2: nbins = st.slider("Bins", 5, 100, 30, key="hist_bins")
            fig = px.histogram(df_c, x=xc, nbins=nbins, color=color_col,
                color_discrete_sequence=COLORS, title=f"Histogram: {xc}", marginal="box")

        elif chart_type == "Box Plot":
            a1,a2 = st.columns(2)
            with a1: yc = ax_select("Y (values)", num_df_cols or all_df_cols, "box_y")
            with a2: xc = ax_select("X (groups)", ["None"]+all_df_cols, "box_x")
            xc = None if xc == "None" else xc
            fig = px.box(df_c, x=xc, y=yc, color=color_col or xc,
                color_discrete_sequence=COLORS, title=f"Box Plot: {yc}", points="all")

        elif chart_type == "Pie Chart":
            a1,a2 = st.columns(2)
            with a1: names_c = ax_select("Labels", all_df_cols, "pie_names")
            with a2:
                val_opts = ["Count"]+num_df_cols
                val_c = st.selectbox("Values", val_opts, key="pie_vals")
            if val_c == "Count":
                pie_data = df_c[names_c].value_counts().reset_index()
                pie_data.columns = [names_c, "Count"]
                fig = px.pie(pie_data, names=names_c, values="Count",
                    color_discrete_sequence=COLORS, title=f"Pie: {names_c} distribution", hole=0.35)
            else:
                fig = px.pie(df_c, names=names_c, values=val_c,
                    color_discrete_sequence=COLORS, title=f"Pie: {val_c} by {names_c}", hole=0.35)
            if fig: fig.update_traces(textposition='inside', textinfo='percent+label')

        elif chart_type == "Violin Plot":
            a1,a2 = st.columns(2)
            with a1: yc = ax_select("Y (values)", num_df_cols or all_df_cols, "vio_y")
            with a2: xc = ax_select("X (groups)", ["None"]+all_df_cols, "vio_x")
            xc = None if xc == "None" else xc
            fig = px.violin(df_c, x=xc, y=yc, color=color_col or xc,
                color_discrete_sequence=COLORS, box=True, points="all",
                title=f"Violin: {yc}")

        elif chart_type == "Line Chart":
            a1,a2 = st.columns(2)
            with a1: xc = ax_select("X Axis", all_df_cols, "line_x")
            with a2: yc = ax_select("Y Axis", num_df_cols or all_df_cols, "line_y", exclude=xc)
            fig = px.line(df_c.sort_values(xc), x=xc, y=yc, color=color_col,
                color_discrete_sequence=COLORS, title=f"Line: {xc} vs {yc}", markers=True)

        elif chart_type == "Area Chart":
            a1,a2 = st.columns(2)
            with a1: xc = ax_select("X Axis", all_df_cols, "area_x")
            with a2: yc = ax_select("Y Axis", num_df_cols or all_df_cols, "area_y", exclude=xc)
            fig = px.area(df_c.sort_values(xc), x=xc, y=yc, color=color_col,
                color_discrete_sequence=COLORS, title=f"Area: {xc} vs {yc}")

        elif chart_type == "Density (KDE)":
            xc = ax_select("Column", num_df_cols or all_df_cols, "kde_x")
            fig = px.histogram(df_c, x=xc, color=color_col, marginal="rug",
                histnorm="density", color_discrete_sequence=COLORS,
                title=f"KDE Density: {xc}")

        elif chart_type == "ECDF Plot":
            xc = ax_select("Column", num_df_cols or all_df_cols, "ecdf_x")
            fig = px.ecdf(df_c, x=xc, color=color_col,
                color_discrete_sequence=COLORS, title=f"ECDF: {xc}")

        elif chart_type == "Regression Plot":
            a1,a2 = st.columns(2)
            with a1: xc = ax_select("X Axis", num_df_cols or all_df_cols, "reg_x")
            with a2: yc = ax_select("Y Axis", num_df_cols or all_df_cols, "reg_y", exclude=xc)
            fig = px.scatter(df_c, x=xc, y=yc, color=color_col,
                color_discrete_sequence=COLORS, trendline="ols",
                title=f"Regression: {xc} vs {yc}")

        elif chart_type == "Heatmap (Correlation)":
            num_only = df_c.select_dtypes(include=[np.number])
            if len(num_only.columns) < 2:
                st.info("Need ≥2 numeric columns for correlation heatmap.")
            else:
                corr = num_only.corr()
                fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                    colorscale=[[0,"#ff6b6b"],[.5,"#1a1a26"],[1,"#7c5cfc"]],
                    text=np.round(corr.values,2), texttemplate="%{text}",
                    colorbar=dict(title="Correlation")))
                fig.update_layout(**PT, title="Correlation Heatmap")

        elif chart_type == "Compare Plot":
            st.markdown("##### Compare multiple columns side by side")
            comp_cols = st.multiselect("Select columns to compare", num_df_cols, default=num_df_cols[:3] if len(num_df_cols)>=3 else num_df_cols, key="comp_cols")
            comp_type = st.radio("Compare as", ["Box","Violin","Bar (mean)"], horizontal=True, key="comp_type")
            if comp_cols:
                df_melt = df_c[comp_cols].melt(var_name="Column", value_name="Value")
                if comp_type == "Box":
                    fig = px.box(df_melt, x="Column", y="Value", color="Column",
                        color_discrete_sequence=COLORS, title="Compare: Box Plot", points="all")
                elif comp_type == "Violin":
                    fig = px.violin(df_melt, x="Column", y="Value", color="Column",
                        color_discrete_sequence=COLORS, title="Compare: Violin Plot", box=True)
                else:
                    mean_df = df_c[comp_cols].mean().reset_index(); mean_df.columns=["Column","Mean"]
                    fig = px.bar(mean_df, x="Column", y="Mean", color="Column",
                        color_discrete_sequence=COLORS, title="Compare: Mean Values", text="Mean")
                    if fig: fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

        if fig:
            fig.update_layout(**PT)
            st.plotly_chart(fig, use_container_width=True)

    render_step_navigation(
        back_step=7,
        back_label="◀ Back to Training",
        next_step=9,
        next_label="Next → Report & Export ▶"
    )
    footer()


# STEP 9 — REPORT & EXPORT (professional Excel + PDF + Word)

elif st.session_state.step == 9:
    if st.session_state.df_clean is None:
        render_empty_state(
            title="No Session Data Available for Export",
            description="Please upload a dataset and train models before generating executive reports and downloading model artifacts.",
            cta_label="Go to Step 1 (Data Ingestion)",
            cta_step=1
        )
        st.stop()

    sh("", "Report & Export", "Download your complete ML session report — Excel, PDF & Word")

    df_clean     = st.session_state.df_clean
    results_df   = st.session_state.results_df
    best_name    = st.session_state.best_model_name
    problem_type = st.session_state.problem_type
    cluster_res  = st.session_state.cluster_results
    ai_report    = st.session_state.ai_report or {}

    st.markdown("""<div class="ai-card">
    <strong>AI Report Guide:</strong>
    Select sections to include below. All 3 export formats provide structured summaries with customized layouts.
    <strong>Excel</strong> = Multi-tab spreadsheet with styled tables and embedded charts.
    <strong>PDF</strong> = Print-ready document with formatted tables and telemetry summaries.
    <strong>Word</strong> = Fully editable document suitable for stakeholder delivery.
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Select Sections to Include")
    OPTS = {
        "Dataset Overview":               df_clean is not None,
        "Column Profiling Summary":       df_clean is not None,
        "Missing Value Handling Log":     st.session_state.missing_applied,
        "Duplicate Removal Log":          st.session_state.dup_removed,
        "Preprocessing Steps":            len(st.session_state.preprocessing_log) > 0,
        "Clean Dataset":                  df_clean is not None,
        "Model Training Results":         results_df is not None,
        "Clustering Silhouette Scores":   cluster_res is not None,
        "AI Recommendations":             True,
    }
    selected = {}
    cols_c = st.columns(2)
    for i,(lbl,avail) in enumerate(OPTS.items()):
        with cols_c[i%2]:
            if avail:
                selected[lbl] = st.checkbox(lbl, value=True, key=f"chk_{i}")
            else:
                st.checkbox(lbl, value=False, disabled=True, key=f"chkd_{i}",
                            help="Step not completed or All ready available")
                selected[lbl] = False

    st.markdown("---")

    #  Equal-height format cards 
    CARD_H = "170px"
    fc1, fc2, fc3 = st.columns(3)

    for col, fmt_badge, fmt_title, fmt_desc in [
        (fc1, "XLSX", "Excel Report", "Multi-sheet · Colored tables · Charts · AI Guide"),
        (fc2, "PDF",  "PDF Report",   "Professional · Color sections · Print-ready · AI insights"),
        (fc3, "DOCX", "Word Report",  "Editable .docx · Colored tables · Section headers"),
    ]:
        with col:
            st.markdown(f"""<div class="glass-card" style="min-height:{CARD_H};margin-bottom:1rem;text-align:center">
            <div style="margin-bottom:0.75rem"><span class="badge badge-primary" style="font-size:0.75rem;padding:0.25rem 0.6rem;font-weight:700">{fmt_badge}</span></div>
            <div style="font-weight:700;color:var(--text);font-size:1.05rem">{fmt_title}</div>
            <div style="font-size:.76rem;color:var(--muted);margin-top:.4rem;line-height:1.5">{fmt_desc}</div>
            </div>""", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)

    # ════ EXCEL ══════════════════════════════════════════════
    with fc1:
        if st.button("Download Excel (.xlsx)", use_container_width=True):
            try:
                import datetime
                from openpyxl import load_workbook
                from openpyxl.styles import (PatternFill, Font, Alignment, Border, Side, GradientFill)
                from openpyxl.utils import get_column_letter
                from openpyxl.chart import BarChart, Reference
                from openpyxl.chart.series import SeriesLabel

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    def ws(df_t, name):
                        df_t.to_excel(w, sheet_name=name[:31], index=False)

                    # Always write cover
                    cover_data = [["MLForge — Session Report", ""],
                                  ["Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
                                  ["Rows", f"{len(df_clean):,}" if df_clean is not None else "N/A"],
                                  ["Columns", str(len(df_clean.columns)) if df_clean is not None else "N/A"],
                                  ["Best Model", best_name or "N/A"],
                                  ["Task Type", problem_type or "N/A"],
                                  ["Training Dataset", st.session_state.get("training_dataset", "preprocessed").title()]]
                    ws(pd.DataFrame(cover_data, columns=["Field","Value"]), "Cover")

                    if selected.get("Dataset Overview") and df_clean is not None:
                        info = get_basic_info(df_clean)
                        ws(pd.DataFrame([
                            ["Total Rows", f"{info['rows']:,}"],
                            ["Total Columns", str(info["columns"])],
                            ["Missing (post-clean)", str(int(df_clean.isna().sum().sum()))],
                            ["Duplicates (original)", str(info["duplicate_rows"])],
                            ["Best Model", best_name or "N/A"],
                            ["Task Type", problem_type or "N/A"],
                        ], columns=["Metric","Value"]), "Overview")

                    if selected.get("Column Profiling Summary") and df_clean is not None:
                        ws(get_column_summary(df_clean), "Column Summary")
                        ns = get_numeric_stats(df_clean)
                        if not ns.empty: ws(ns, "Numeric Stats")
                        
                        # Add correlation matrix
                        numeric_cols = df_clean.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 1:
                            corr = df_clean[numeric_cols].corr()
                            # Reset index to include column names in the sheet
                            corr_export = corr.reset_index().rename(columns={'index': 'Feature'})
                            ws(corr_export, "Correlation Matrix")

                    if selected.get("Preprocessing Steps") and st.session_state.preprocessing_log:
                        ws(pd.DataFrame({"Step": range(1, len(st.session_state.preprocessing_log)+1),
                                         "Action": st.session_state.preprocessing_log}), "Preprocessing")

                    if selected.get("Missing Value Handling Log") and st.session_state.missing_applied:
                        ws(pd.DataFrame([{"Metric": "Strategy", "Value": "AI-Driven Imputation"},
                                         {"Metric": "Applied", "Value": "Yes"}]), "Missing Handling")

                    if selected.get("Duplicate Removal Log") and st.session_state.dup_removed:
                        ws(pd.DataFrame([{"Metric": "Duplicates Removed", "Value": "Yes"},
                                         {"Metric": "Status", "Value": "Cleaned"}]), "Duplicate Removal")

                    if selected.get("Clean Dataset") and df_clean is not None:
                        ws(df_clean.head(100), "Clean Data")

                    if selected.get("Model Training Results") and results_df is not None:
                        ws(results_df, "Model Results")

                    if selected.get("Clustering Silhouette Scores") and cluster_res:
                        ws(pd.DataFrame([{"Algorithm":k,"Silhouette Score":v["score"]}
                                          for k,v in cluster_res.items()]), "Clustering")

                    if selected.get("AI Recommendations"):
                        rows = [["Project Objective", (problem_type or "N/A").upper()],
                                ["Target Metric", "F1/R2 Score (Maximization)"],
                                ["", ""],
                                ["RECOMMENDED ACTION", f"Deploy {best_name or 'the top model'} for production use."]]
                        for m in ai_report.get("recommended_models",[]): rows.append(["Suggested Model", m])
                        for f in ai_report.get("important_features",[]): rows.append(["Key Feature", str(f)])
                        cl = ai_report.get("clustering",{})
                        if cl.get("recommended"): rows.append(["Cluster Strategy", f"{cl['recommended']} — {cl.get('reason','')}"])
                        rows.append(["", ""])
                        rows.append(["NEXT STEPS", "1. Download the Pickle/ONNX model files."])
                        rows.append(["", "2. Integrate the model into your application pipeline."])
                        rows.append(["", "3. Monitor model drift with fresh data over time."])
                        if rows: ws(pd.DataFrame(rows, columns=["Insight Category","Actionable Guidance"]), "AI Guide")

                # Apply ultra-pro styling
                buf.seek(0)
                wb = load_workbook(buf)

                # Color palette
                C_NAVY  = "1E3A5F"
                C_TEAL  = "00D4AA"
                C_PURP  = "7C5CFC"
                C_GOLD  = "FFB347"
                C_ALT1  = "EEF2FF"
                C_ALT2  = "E8FFF9"
                C_ALT3  = "FFF8EE"

                # Sheet-specific accent colors
                SHEET_COLORS = {
                    "Cover":             (C_NAVY, C_ALT1, "1E3A5F"),
                    "Overview":          (C_NAVY, C_ALT1, "1E3A5F"),
                    "Column Summary":    (C_PURP, C_ALT1, "7C5CFC"),
                    "Numeric Stats":     (C_TEAL, C_ALT2, "00608A"),
                    "Correlation Matrix":(C_PURP, C_ALT1, "7C5CFC"),
                    "Preprocessing":     (C_PURP, C_ALT1, "7C5CFC"),
                    "Missing Handling":  (C_TEAL, C_ALT2, "00608A"),
                    "Duplicate Removal": (C_PURP, C_ALT1, "7C5CFC"),
                    "Clean Data":        (C_NAVY, C_ALT1, "1E3A5F"),
                    "Model Results":     (C_NAVY, C_ALT1, "1E3A5F"),
                    "Clustering":        (C_TEAL, C_ALT2, "00608A"),
                    "AI Guide":          (C_GOLD, C_ALT3, "B8860B"),
                }

                thin = Side(style="thin", color="C0C0D8")
                BDR = Border(left=thin, right=thin, top=thin, bottom=thin)

                for ws_obj in wb.worksheets:
                    sname = ws_obj.title
                    hdr_c, alt_c, tab_c = SHEET_COLORS.get(sname, (C_NAVY, C_ALT1, "1E3A5F"))
                    HFill  = PatternFill("solid", fgColor=hdr_c)
                    AFill  = PatternFill("solid", fgColor=alt_c)
                    WHFill = PatternFill("solid", fgColor="FFFFFF")
                    WFont  = Font(color="FFFFFF", bold=True, name="Calibri", size=10)
                    HFont  = Font(color=hdr_c, bold=True, name="Calibri", size=9)
                    DFont  = Font(color="1E1E2E", name="Calibri", size=9)

                    # Auto-width
                    for col_cells in ws_obj.columns:
                        max_l = max((len(str(c.value or "")) for c in col_cells), default=8)
                        ws_obj.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_l + 4, 50)

                    # Style rows
                    for ri, row in enumerate(ws_obj.iter_rows()):
                        for cell in row:
                            cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
                            if ri == 0:
                                cell.fill = HFill; cell.font = WFont; cell.border = BDR
                                ws_obj.row_dimensions[ri+1].height = 24
                            elif ri % 2 == 1:
                                cell.fill = AFill; cell.font = DFont; cell.border = BDR
                            else:
                                cell.fill = WHFill; cell.font = DFont; cell.border = BDR

                    ws_obj.freeze_panes = "A2"
                    ws_obj.auto_filter.ref = ws_obj.dimensions
                    ws_obj.sheet_properties.tabColor = tab_c

                #  Interactive Hyperlinks on Cover 
                if "Cover" in wb.sheetnames:
                    ws_c = wb["Cover"]
                    
                    # Merge and style the title cell
                    ws_c.merge_cells("A2:B2")
                    title_cell = ws_c["A2"]
                    title_cell.font = Font(color="FFFFFF", bold=True, name="Calibri", size=12)
                    title_cell.fill = PatternFill("solid", fgColor="1E3A5F")
                    title_cell.alignment = Alignment(horizontal="center", vertical="center")
                    
                    # Add dynamic links to all generated sheets
                    link_idx = 10
                    for sname in wb.sheetnames:
                        if sname != "Cover":
                            cell = ws_c.cell(row=link_idx, column=1, value=f"Go to {sname}")
                            cell.hyperlink = f"#'{sname}'!A1"
                            cell.font = Font(color="0000FF", underline="single", name="Calibri", size=10)
                            cell.alignment = Alignment(horizontal="left")
                            link_idx += 1

                # Data Bars for Model Results
                if "Model Results" in wb.sheetnames and results_df is not None:
                    from openpyxl.formatting.rule import DataBarRule
                    ws_m = wb["Model Results"]
                    n_rows = len(results_df) + 1
                    n_cols = len(results_df.columns)
                    # Apply Data Bars to the primary metric column
                    target_col = 2 # Usually Accuracy or F1
                    col_letter = get_column_letter(target_col)
                    ws_m.conditional_formatting.add(f"{col_letter}2:{col_letter}{n_rows}", 
                        DataBarRule(start_type="min", end_type="max", color="638EC6", showValue=True, minLength=None, maxLength=None))

                # Conditional formatting on Model Results
                if "Model Results" in wb.sheetnames and results_df is not None:
                    from openpyxl.formatting.rule import ColorScaleRule
                    ws_m = wb["Model Results"]
                    n_rows = len(results_df) + 1
                    n_cols = len(results_df.columns)
                    for ci in range(2, n_cols + 1):
                        col_letter = get_column_letter(ci)
                        rng = f"{col_letter}2:{col_letter}{n_rows}"
                        ws_m.conditional_formatting.add(rng, ColorScaleRule(
                            start_type="min", start_color="FF6B6B",
                            mid_type="percentile", mid_value=50, mid_color="FFEB84",
                            end_type="max", end_color="00D4AA"
                        ))
                    # Bar chart
                    try:
                        chart = BarChart()
                        chart.type = "col"; chart.grouping = "clustered"
                        chart.title = "Model Performance Comparison"
                        chart.style = 10
                        data_ref = Reference(ws_m, min_col=2, max_col=n_cols, min_row=1, max_row=n_rows)
                        cats_ref = Reference(ws_m, min_col=1, min_row=2, max_row=n_rows)
                        chart.add_data(data_ref, titles_from_data=True)
                        chart.set_categories(cats_ref)
                        chart.width = 25; chart.height = 12
                        ws_m.add_chart(chart, f"A{n_rows+3}")
                    except Exception: pass

                # Clustering chart
                if "Clustering" in wb.sheetnames and cluster_res:
                    try:
                        chart_c = BarChart()
                        chart_c.type = "col"
                        chart_c.title = "Silhouette Scores by Algorithm"
                        chart_c.style = 10
                        ws_cl = wb["Clustering"]
                        n_rows_cl = len(cluster_res) + 1
                        data_ref = Reference(ws_cl, min_col=2, max_col=2, min_row=1, max_row=n_rows_cl)
                        cats_ref = Reference(ws_cl, min_col=1, min_row=2, max_row=n_rows_cl)
                        chart_c.add_data(data_ref, titles_from_data=True)
                        chart_c.set_categories(cats_ref)
                        chart_c.width = 15; chart_c.height = 10
                        ws_cl.add_chart(chart_c, f"A{n_rows_cl+3}")
                    except Exception: pass

                # Heatmap for Correlation Matrix
                if "Correlation Matrix" in wb.sheetnames:
                    from openpyxl.formatting.rule import ColorScaleRule
                    ws_c = wb["Correlation Matrix"]
                    n_rows = ws_c.max_row; n_cols = ws_c.max_column
                    # Symmetric color scale for correlation (-1 to 1)
                    rng = f"B2:{get_column_letter(n_cols)}{n_rows}"
                    ws_c.conditional_formatting.add(rng, ColorScaleRule(
                        start_type="num", start_value=-1, start_color="FF6B6B",
                        mid_type="num", mid_value=0, mid_color="FFFFFF",
                        end_type="num", end_value=1, end_color="00D4AA"
                    ))

                styled = io.BytesIO(); wb.save(styled); styled.seek(0)
                st.download_button("MLForge_Report.xlsx", data=styled.getvalue(),
                    file_name="MLForge_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.success("Excel report ready.")
            except Exception as e:
                st.error(f"Excel error: {e}")
                with st.expander("Debug"): st.code(traceback.format_exc())

    # ════ PDF ════════════════════════════════════════════════
    with fc2:
        if st.button("Download PDF (.pdf)", use_container_width=True):
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.lib import colors
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import cm
                from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                                Table, TableStyle, HRFlowable, KeepTogether,
                                                PageBreak)
                from reportlab.lib.enums import TA_CENTER, TA_LEFT
                from reportlab.graphics.shapes import Drawing, Rect, String
                from reportlab.graphics.charts.barcharts import VerticalBarChart
                from reportlab.graphics import renderPDF

                W, H = A4
                NAVY  = colors.HexColor("#1E3A5F")
                PRP   = colors.HexColor("#7c5cfc")
                TEL   = colors.HexColor("#00d4aa")
                YEL   = colors.HexColor("#ffb347")
                RED   = colors.HexColor("#ff6b6b")
                GRY   = colors.HexColor("#666680")
                BGLT  = colors.HexColor("#EEF2FF")
                BGTL  = colors.HexColor("#E8FFF9")
                WHT   = colors.white
                DRK   = colors.HexColor("#1E1E2E")

                pdf_buf = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buf, pagesize=A4,
                    leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=2*cm, bottomMargin=2*cm)
                S = getSampleStyleSheet()

                sT   = ParagraphStyle("T",  textColor=NAVY,  fontSize=28, leading=34, fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4)
                sSB  = ParagraphStyle("SB", textColor=TEL,   fontSize=11, leading=14, fontName="Helvetica",      alignment=TA_CENTER, spaceAfter=16)
                sH2  = ParagraphStyle("H2", textColor=WHT,   fontSize=12, leading=14, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=0,
                                      backColor=NAVY, leftIndent=0, rightIndent=0, borderPad=7)
                sBD  = ParagraphStyle("BD", fontSize=9,  leading=12, fontName="Helvetica", textColor=DRK, spaceAfter=3, leftIndent=10)
                sMT  = ParagraphStyle("MT", fontSize=8,  leading=10, fontName="Helvetica", textColor=GRY, spaceAfter=2, leftIndent=10)

                def sec_hdr(txt, color=NAVY):
                    bg = ParagraphStyle("sh", textColor=WHT, fontSize=14, leading=16, fontName="Helvetica-Bold",
                                        spaceBefore=16, spaceAfter=6, backColor=color,
                                        leftIndent=0, rightIndent=0, borderPad=10)
                    return Paragraph(f"  {txt}", bg)

                def make_table(data, cw=None, hdr=NAVY, alt=BGLT):
                    t = Table(data, colWidths=cw, repeatRows=1, hAlign='LEFT')
                    t.setStyle(TableStyle([
                        ("BACKGROUND",    (0,0), (-1,0), hdr),
                        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHT, alt]),
                        ("TEXTCOLOR",     (0,0), (-1,0), WHT),
                        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                        ("FONTSIZE",      (0,0), (-1,0), 11),
                        ("FONTSIZE",      (0,1), (-1,-1), 9),
                        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
                        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                        ("PADDING",       (0,0), (-1,-1), 6),
                        ("TOPPADDING",    (0,0), (-1,0), 10),
                        ("BOTTOMPADDING", (0,0), (-1,0), 10),
                        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#B0B8E0")),
                    ]))
                    return t

                def bar_chart_pdf(labels, values, title="", color=PRP):
                    drawing = Drawing(460, 220)
                    bc = VerticalBarChart()
                    bc.x = 50; bc.y = 60; bc.width = 380; bc.height = 130
                    bc.data = [values]
                    bc.bars[0].fillColor = color
                    # Truncate long labels so they don't overlap
                    bc.categoryAxis.categoryNames = [str(L)[:15]+".." if len(str(L))>16 else str(L) for L in labels]
                    bc.categoryAxis.labels.angle = 45
                    bc.categoryAxis.labels.dx = 0
                    bc.categoryAxis.labels.dy = -10
                    bc.categoryAxis.labels.fontSize = 7
                    bc.valueAxis.labels.fontSize = 7
                    
                    # Adjust y-axis range
                    v_min = min(values) if values else 0
                    v_max = max(values) if values else 1
                    bc.valueAxis.valueMin = v_min * 1.1 if v_min < 0 else 0
                    bc.valueAxis.valueMax = v_max * 1.2
                    
                    bc.strokeColor = colors.HexColor("#B0B8E0")
                    
                    # Value labels on top of bars
                    bc.barLabelFormat = "%.2f"
                    bc.barLabels.nudge = 5
                    bc.barLabels.fontSize = 6
                    bc.barLabels.boxAnchor = 's'
                    
                    drawing.add(bc)
                    drawing.add(String(230, 205, title, textAnchor='middle',
                                       fontSize=10, fillColor=NAVY, fontName="Helvetica-Bold"))
                    return drawing


                import datetime as _dt

                # Page-number canvas maker
                class _PageNum:
                    def __init__(self, doc):
                        self.doc = doc
                    def afterPage(self):
                        pass
                def _add_page_num(canvas, doc):
                    canvas.saveState()
                    canvas.setFont("Helvetica", 8)
                    canvas.setFillColor(GRY)
                    canvas.drawCentredString(W/2, 1.2*cm, f"MLForge — Page {doc.page}")
                    canvas.restoreState()

                # Cover page elements
                _now = _dt.datetime.now().strftime("%B %d, %Y  %H:%M")
                _rows = len(df_clean) if df_clean is not None else 0
                _cols = len(df_clean.columns) if df_clean is not None else 0

                # Large cover block
                _cover_table_data = [["MLForge", "Session Report"],
                                     ["Generated", _now],
                                     ["Dataset", f"{_rows:,} rows × {_cols} columns"],
                                     ["Best Model", best_name or "N/A"],
                                     ["Task Type", (problem_type or "N/A").title()]]
                _ct = Table(_cover_table_data, colWidths=[6*cm, 11.4*cm], hAlign='LEFT')
                _ct.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), NAVY),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#EEF2FF"), colors.white]),
                    ("TEXTCOLOR",  (0,0), (-1,0), TEL),
                    ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                    ("FONTSIZE",   (0,0), (-1,0), 16),
                    ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#EEF2FF")),
                    ("TEXTCOLOR",  (0,1), (-1,-1), colors.HexColor("#1E1E2E")),
                    ("FONTNAME",   (0,1), (0,-1), "Helvetica-Bold"),
                    ("FONTNAME",   (1,1), (1,-1), "Helvetica"),
                    ("FONTSIZE",   (0,1), (-1,-1), 10),
                    ("ALIGN",      (0,0), (-1,-1), "LEFT"),
                    ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
                    ("PADDING",    (0,0), (-1,-1), 10),
                    ("TOPPADDING", (0,0), (-1,0), 12),
                    ("BOTTOMPADDING",(0,0),(-1,0), 12),
                    ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#C0C0D8")),
                ]))

                story = [
                    Spacer(1, 1.5*cm),
                    Paragraph("MLForge", sT),
                    Paragraph("Machine Learning Session Report", sSB),
                    HRFlowable(width="100%", color=NAVY, thickness=4),
                    Spacer(1, 0.8*cm),
                    _ct,
                    Spacer(1, 0.5*cm),
                    HRFlowable(width="100%", color=TEL, thickness=1),
                    Spacer(1, 0.5*cm),
                ]

                if selected.get("Dataset Overview") and df_clean is not None:
                    info = get_basic_info(df_clean)
                    story += [sec_hdr("Dataset Overview"), Spacer(1,4)]
                    data = [["Metric","Value"],
                            ["Total Rows", f"{info['rows']:,}"],
                            ["Total Columns", str(info["columns"])],
                            ["Missing (post-clean)", str(int(df_clean.isna().sum().sum()))],
                            ["Best Model", best_name or "N/A"],
                            ["Task Type", problem_type or "N/A"]]
                    story += [make_table(data, cw=[8.5*cm,8.9*cm]), Spacer(1,10)]

                if selected.get("Column Profiling Summary") and df_clean is not None:
                    story += [sec_hdr("Column Profiling"), Spacer(1,4)]
                    cs = get_column_summary(df_clean)
                    hdr_r = list(cs.columns)
                    rows_r = [[str(v) for v in r] for _,r in cs.head(15).iterrows()]
                    cw_r = [(W-3.6*cm)/len(hdr_r)]*len(hdr_r)
                    story += [make_table([hdr_r]+rows_r, cw=cw_r, alt=BGTL, hdr=PRP), Spacer(1,10)]
                    
                    ns = get_numeric_stats(df_clean)
                    if not ns.empty:
                        story += [sec_hdr("Numeric Stats", color=colors.HexColor("#00608A")), Spacer(1,4)]
                        hdr_ns = list(ns.columns)
                        rows_ns = [[str(round(v,4) if isinstance(v,float) else v) for v in r] for _,r in ns.head(15).iterrows()]
                        cw_ns = [(W-3.6*cm)/len(hdr_ns)]*len(hdr_ns)
                        story += [make_table([hdr_ns]+rows_ns, cw=cw_ns, alt=colors.HexColor("#E8FFF9"), hdr=colors.HexColor("#00608A")), Spacer(1,10)]
                    
                    numeric_cols = df_clean.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 1:
                        corr = df_clean[numeric_cols].corr()
                        corr_export = corr.reset_index().rename(columns={'index': 'Feature'})
                        story += [sec_hdr("Correlation Matrix", color=PRP), Spacer(1,4)]
                        hdr_c = list(corr_export.columns)
                        rows_c = [[str(round(v,4) if isinstance(v,float) else v)[:8] for v in r] for _,r in corr_export.iterrows()]
                        cw_c = [(W-3.6*cm)/len(hdr_c)]*len(hdr_c)
                        story += [make_table([hdr_c]+rows_c, cw=cw_c, alt=BGTL, hdr=PRP), Spacer(1,10)]

                if selected.get("Preprocessing Steps") and st.session_state.preprocessing_log:
                    story += [sec_hdr("Preprocessing Steps"), Spacer(1,4)]
                    pdata = [["#","Action"]] + [[str(i+1), step] for i,step in enumerate(st.session_state.preprocessing_log)]
                    story += [make_table(pdata, cw=[1.5*cm, 13*cm], hdr=PRP), Spacer(1,10)]

                if selected.get("Missing Value Handling Log") and st.session_state.missing_applied:
                    story += [sec_hdr("Missing Handling", color=colors.HexColor("#00608A")), Spacer(1,4)]
                    mdata = [["Metric", "Value"], ["Strategy", "AI-Driven Imputation"], ["Applied", "Yes"]]
                    story += [make_table(mdata, cw=[6.5*cm, 10.9*cm], hdr=colors.HexColor("#00608A")), Spacer(1,10)]

                if selected.get("Duplicate Removal Log") and st.session_state.dup_removed:
                    story += [sec_hdr("Duplicate Removal", color=PRP), Spacer(1,4)]
                    ddata = [["Metric", "Value"], ["Duplicates Removed", "Yes"], ["Status", "Cleaned"]]
                    story += [make_table(ddata, cw=[6.5*cm, 10.9*cm], hdr=PRP), Spacer(1,10)]
                    
                if selected.get("Clean Dataset") and df_clean is not None:
                    story += [sec_hdr("Clean Data (First 30 rows)", color=NAVY), Spacer(1,4)]
                    cd_df = df_clean.head(30)
                    hdr_cd = list(cd_df.columns)
                    rows_cd = [[str(round(v,4) if isinstance(v,float) else v)[:12] for v in r] for _,r in cd_df.iterrows()]
                    cw_cd = [(W-3.6*cm)/len(hdr_cd)]*len(hdr_cd)
                    story += [make_table([hdr_cd]+rows_cd, cw=cw_cd, alt=BGLT, hdr=NAVY), Spacer(1,10)]

                if selected.get("Model Training Results") and results_df is not None:
                    story += [sec_hdr("Model Training Results"), Spacer(1,4)]
                    hdr_r = list(results_df.columns)
                    rows_r = [[str(round(v,4) if isinstance(v,float) else v) for v in r] for _,r in results_df.iterrows()]
                    cw_r = [(W-3.6*cm)/len(hdr_r)]*len(hdr_r)
                    story += [make_table([hdr_r]+rows_r, cw=cw_r), Spacer(1,6)]
                    story.append(Paragraph(f"Best Model: {best_name}", sBD))
                    # Add bar chart
                    num_res = results_df.select_dtypes("number").columns.tolist()
                    if num_res and len(results_df) > 0:
                        metric_c = "F1 Score" if "F1 Score" in num_res else (num_res[0] if num_res else None)
                        if metric_c:
                            story += [Spacer(1,8),
                                      bar_chart_pdf(results_df["Model"].tolist(),
                                                    [float(v) for v in results_df[metric_c].tolist()],
                                                    title=f"{metric_c} by Model"),
                                      Spacer(1,10)]

                if selected.get("Clustering Silhouette Scores") and cluster_res:
                    story += [sec_hdr("Clustering Results", color=colors.HexColor("#00608A")), Spacer(1,4)]
                    cdata = [["Algorithm","Silhouette Score"]] + [[k, str(v["score"])] for k,v in cluster_res.items()]
                    story += [make_table(cdata, cw=[8.5*cm,8.9*cm], hdr=colors.HexColor("#00608A"), alt=BGTL),
                              Spacer(1,8)]
                    # Cluster bar chart
                    cl_labels = list(cluster_res.keys())
                    cl_scores = [float(v["score"]) for v in cluster_res.values()]
                    story += [bar_chart_pdf(cl_labels, cl_scores, "Silhouette Scores by Algorithm"), Spacer(1,10)]

                if selected.get("AI Recommendations"):
                    story += [sec_hdr("AI Recommendations", color=colors.HexColor("#B8860B")), Spacer(1,6)]
                    pts = []
                    pts.append(f"<b>Project Objective:</b> {(problem_type or 'N/A').upper()}")
                    pts.append("<b>Target Metric:</b> F1/R2 Score (Maximization)")
                    pts.append("")
                    pts.append(f"<b>RECOMMENDED ACTION:</b> Deploy {best_name or 'the top model'} for production use.")
                    for m in ai_report.get("recommended_models",[]): pts.append(f"&bull; <b>Suggested Model:</b> {m}")
                    for f in ai_report.get("important_features",[]): pts.append(f"&bull; <b>Key Feature:</b> {f}")
                    cl = ai_report.get("clustering",{})
                    if cl.get("recommended"): pts.append(f"&bull; <b>Clustering Strategy:</b> {cl['recommended']} &mdash; {cl.get('reason','')}")
                    pts.append("")
                    pts.append("<b>NEXT STEPS:</b>")
                    pts.append("1. Download the Pickle/ONNX model files.")
                    pts.append("2. Integrate the model into your application pipeline.")
                    pts.append("3. Monitor model drift with fresh data over time.")
                    for pt in pts:
                        story.append(Paragraph(pt, sBD))
                    story.append(Spacer(1,10))

                doc.build(story, onFirstPage=_add_page_num, onLaterPages=_add_page_num)
                pdf_buf.seek(0)
                st.download_button("MLForge_Report.pdf", data=pdf_buf.getvalue(),
                    file_name="MLForge_Report.pdf", mime="application/pdf")
                st.success("PDF ready.")
            except Exception as e:
                st.error(f"PDF error: {e}")
                with st.expander("Debug"): st.code(traceback.format_exc())

    # ════ WORD ═══════════════════════════════════════════════
    with fc3:
        if st.button("Download Word (.docx)", use_container_width=True):
            try:
                from docx import Document
                from docx.shared import Pt, RGBColor, Cm, Inches
                from docx.enum.text import WD_ALIGN_PARAGRAPH
                from docx.oxml.ns import qn
                from docx.oxml import OxmlElement

                dw = Document()
                for sec in dw.sections:
                    sec.top_margin=Cm(2); sec.bottom_margin=Cm(2)
                    sec.left_margin=Cm(2.5); sec.right_margin=Cm(2.5)

                def set_cell_bg(cell, hex_c):
                    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
                    shd = OxmlElement('w:shd')
                    shd.set(qn('w:val'),'clear'); shd.set(qn('w:color'),'auto')
                    shd.set(qn('w:fill'), hex_c); tcPr.append(shd)

                def add_para_border(para, hex_c="1E3A5F", sz=24):
                    """Add a bottom border to a paragraph."""
                    from docx.oxml import OxmlElement
                    pPr = para._p.get_or_add_pPr()
                    pBdr = OxmlElement('w:pBdr')
                    bottom = OxmlElement('w:bottom')
                    bottom.set(qn('w:val'), 'single')
                    bottom.set(qn('w:sz'), str(sz))
                    bottom.set(qn('w:color'), hex_c)
                    pBdr.append(bottom)
                    pPr.append(pBdr)

                import datetime as _dt2
                _now_w = _dt2.datetime.now().strftime("%B %d, %Y  %H:%M")

                #  Cover page 
                # Big gradient-like header using a 1-row table
                cover_tb = dw.add_table(rows=1, cols=1)
                cover_tb.style = "Table Grid"
                cover_cell = cover_tb.rows[0].cells[0]
                set_cell_bg(cover_cell, "1E3A5F")
                cover_cell.width = Cm(16)
                cp1 = cover_cell.paragraphs[0]
                cp1.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cr1 = cp1.add_run("MLForge")
                cr1.font.size = Pt(28); cr1.font.bold = True
                cr1.font.color.rgb = RGBColor(0x00,0xD4,0xAA)
                cp2 = cover_cell.add_paragraph()
                cp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cr2 = cp2.add_run("Machine Learning Session Report")
                cr2.font.size = Pt(13); cr2.font.color.rgb = RGBColor(0xFF,0xFF,0xFF)

                dw.add_paragraph()

                # KPI info table on cover
                _cv_rows = len(df_clean) if df_clean is not None else 0
                _cv_cols = len(df_clean.columns) if df_clean is not None else 0
                kpi_data = [["Field","Value"],
                             ["Generated", _now_w],
                             ["Dataset", f"{_cv_rows:,} rows × {_cv_cols} columns"],
                             ["Best Model", best_name or "N/A"],
                             ["Task Type", (problem_type or "N/A").title()],
                             ["Training Dataset", st.session_state.training_dataset.title()]]
                kpi_tb = dw.add_table(rows=len(kpi_data), cols=2)
                kpi_tb.style = "Table Grid"
                for ri, row_d in enumerate(kpi_data):
                    cells = kpi_tb.rows[ri].cells
                    for ci, val in enumerate(row_d):
                        cells[ci].text = str(val)
                        bg = "1E3A5F" if ri == 0 else ("EEF2FF" if ri % 2 == 1 else "FFFFFF")
                        set_cell_bg(cells[ci], bg)
                        for run in cells[ci].paragraphs[0].runs:
                            run.font.size = Pt(10)
                            if ri == 0:
                                run.font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
                                run.font.bold = True
                            elif ci == 0:
                                run.font.bold = True
                                run.font.color.rgb = RGBColor(0x1E,0x3A,0x5F)
                        cells[ci].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT

                dw.add_paragraph()

                def add_sec(icon, title, rgb=(0x1E,0x3A,0x5F)):
                    h = dw.add_heading("", level=1); h.clear()
                    txt = f"{icon}  {title}".strip() if icon else title
                    run = h.add_run(txt); run.font.size=Pt(14); run.font.bold=True
                    run.font.color.rgb = RGBColor(*rgb)

                def add_table(df_t, hdr_hex="1E3A5F", alt_hex="EEF2FF", max_rows=50):
                    df_t = df_t.head(max_rows)
                    tb = dw.add_table(rows=1+len(df_t), cols=len(df_t.columns))
                    tb.style = "Table Grid"
                    for i,col in enumerate(df_t.columns):
                        cell = tb.rows[0].cells[i]
                        cell.text = str(col); set_cell_bg(cell, hdr_hex)
                        for run in cell.paragraphs[0].runs:
                            run.font.color.rgb=RGBColor(0xFF,0xFF,0xFF); run.font.bold=True; run.font.size=Pt(9)
                        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for ri,(_,row) in enumerate(df_t.iterrows()):
                        tr = tb.rows[ri+1]
                        bg = alt_hex if ri%2==0 else "FFFFFF"
                        for i,val in enumerate(row):
                            cell = tr.cells[i]
                            cell.text = str(round(val,4) if isinstance(val,float) else val)
                            set_cell_bg(cell, bg)
                            for run in cell.paragraphs[0].runs:
                                run.font.size=Pt(9); run.font.color.rgb=RGBColor(0x1E,0x1E,0x2E)
                            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    dw.add_paragraph()

                def add_bullet(text, rgb=(0x1E,0x3A,0x5F)):
                    p = dw.add_paragraph(); r = p.add_run(f"•  {text}")
                    r.font.size=Pt(10); r.font.color.rgb=RGBColor(*rgb)

                if selected.get("Dataset Overview") and df_clean is not None:
                    add_sec("","Dataset Overview")
                    info = get_basic_info(df_clean)
                    add_table(pd.DataFrame([
                        ["Total Rows",f"{info['rows']:,}"],["Total Columns",str(info["columns"])],
                        ["Missing (post-clean)",str(int(df_clean.isna().sum().sum()))],
                        ["Best Model",best_name or "N/A"],["Task Type",problem_type or "N/A"],
                    ], columns=["Metric","Value"]), hdr_hex="1E3A5F", alt_hex="EEF2FF")

                if selected.get("Column Profiling Summary") and df_clean is not None:
                    add_sec("","Column Profiling", rgb=(0x00,0x60,0x8A))
                    add_table(get_column_summary(df_clean), hdr_hex="00608A", alt_hex="E8FFF9", max_rows=30)
                    
                    ns = get_numeric_stats(df_clean)
                    if not ns.empty:
                        add_sec("","Numeric Stats", rgb=(0x00,0xD4,0xAA))
                        add_table(ns, hdr_hex="00D4AA", alt_hex="E8FFF9", max_rows=30)
                        
                    numeric_cols = df_clean.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 1:
                        corr = df_clean[numeric_cols].corr()
                        corr_export = corr.reset_index().rename(columns={'index': 'Feature'})
                        add_sec("","Correlation Matrix", rgb=(0x7C,0x5C,0xFC))
                        add_table(corr_export, hdr_hex="7C5CFC", alt_hex="F0EEFF", max_rows=30)

                if selected.get("Preprocessing Steps") and st.session_state.preprocessing_log:
                    add_sec("","Preprocessing Steps", rgb=(0x7c,0x5c,0xfc))
                    add_table(pd.DataFrame({"Step":range(1,len(st.session_state.preprocessing_log)+1),
                                            "Action":st.session_state.preprocessing_log}),
                              hdr_hex="7C5CFC", alt_hex="F0EEFF", max_rows=100)

                if selected.get("Missing Value Handling Log") and st.session_state.missing_applied:
                    add_sec("","Missing Handling", rgb=(0x00,0x60,0x8A))
                    add_table(pd.DataFrame([{"Metric": "Strategy", "Value": "AI-Driven Imputation"},
                                            {"Metric": "Applied", "Value": "Yes"}]),
                              hdr_hex="00608A", alt_hex="E8FFF9")

                if selected.get("Duplicate Removal Log") and st.session_state.dup_removed:
                    add_sec("","Duplicate Removal", rgb=(0x7C,0x5C,0xFC))
                    add_table(pd.DataFrame([{"Metric": "Duplicates Removed", "Value": "Yes"},
                                            {"Metric": "Status", "Value": "Cleaned"}]),
                              hdr_hex="7C5CFC", alt_hex="F0EEFF")

                if selected.get("Clean Dataset") and df_clean is not None:
                    add_sec("","Clean Dataset (first 20 rows)", rgb=(0x5a,0x3f,0xd4))
                    add_table(df_clean.head(20), hdr_hex="5A3FD4", alt_hex="EDE8FF")

                if selected.get("Model Training Results") and results_df is not None:
                    add_sec("","Model Training Results")
                    add_table(results_df, hdr_hex="1E3A5F", alt_hex="EEF2FF")
                    p = dw.add_paragraph(); r = p.add_run(f"Best Model: {best_name}")
                    r.font.bold=True; r.font.size=Pt(11); r.font.color.rgb=RGBColor(0x00,0xD4,0xAA)
                    dw.add_paragraph()
                    try:
                        import matplotlib.pyplot as plt
                        num_res = results_df.select_dtypes("number").columns.tolist()
                        if num_res and len(results_df) > 0:
                            metric_c = "F1 Score" if "F1 Score" in num_res else num_res[0]
                            fig, ax = plt.subplots(figsize=(6.5, 3.5))
                            labels = [str(m)[:15]+".." if len(str(m))>16 else str(m) for m in results_df["Model"]]
                            ax.bar(labels, results_df[metric_c], color="#7C5CFC")
                            ax.set_title(f"{metric_c} by Model", color="#1E3A5F", fontweight='bold')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            img_buf = io.BytesIO()
                            plt.savefig(img_buf, format='png', dpi=150)
                            img_buf.seek(0); plt.close(fig)
                            from docx.shared import Inches
                            dw.add_picture(img_buf, width=Inches(5.5))
                            dw.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                            dw.add_paragraph()
                    except Exception: pass

                if selected.get("Clustering Silhouette Scores") and cluster_res:
                    add_sec("","Clustering Results", rgb=(0x00,0x60,0x8A))
                    add_table(pd.DataFrame([{"Algorithm":k,"Silhouette Score":v["score"]} for k,v in cluster_res.items()]),
                              hdr_hex="00608A", alt_hex="E8FFF9")
                    try:
                        import matplotlib.pyplot as plt
                        cl_labels = list(cluster_res.keys())
                        cl_scores = [float(v["score"]) for v in cluster_res.values()]
                        fig, ax = plt.subplots(figsize=(6.5, 3.5))
                        ax.bar(cl_labels, cl_scores, color="#00608A")
                        ax.set_title("Silhouette Scores by Algorithm", color="#00608A", fontweight='bold')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        img_buf = io.BytesIO()
                        plt.savefig(img_buf, format='png', dpi=150)
                        img_buf.seek(0); plt.close(fig)
                        from docx.shared import Inches
                        dw.add_picture(img_buf, width=Inches(5.5))
                        dw.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                        dw.add_paragraph()
                    except Exception: pass

                if selected.get("AI Recommendations"):
                    add_sec("","AI Recommendations", rgb=(0xB8,0x86,0x0B))
                    
                    pts = []
                    pts.append(f"Project Objective: {(problem_type or 'N/A').upper()}")
                    pts.append("Target Metric: F1/R2 Score (Maximization)")
                    pts.append(f"RECOMMENDED ACTION: Deploy {best_name or 'the top model'} for production use.")
                    for m in ai_report.get("recommended_models",[]): pts.append(f"Suggested Model: {m}")
                    for f in ai_report.get("important_features",[]): pts.append(f"Key Feature: {f}")
                    cl = ai_report.get("clustering",{})
                    if cl.get("recommended"): pts.append(f"Clustering Strategy: {cl['recommended']} — {cl.get('reason','')}")
                    pts.append("NEXT STEPS:")
                    pts.append("1. Download the Pickle/ONNX model files.")
                    pts.append("2. Integrate the model into your application pipeline.")
                    pts.append("3. Monitor model drift with fresh data over time.")
                    for pt in pts:
                        add_bullet(pt)
                    dw.add_paragraph()

                # Add a native footer with page numbers
                for section in dw.sections:
                    doc_footer = section.footer
                    fp = doc_footer.paragraphs[0]
                    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    # Add border top to footer
                    add_para_border(fp, hex_c="B0B8E0", sz=12)
                    fr = fp.add_run("Generated by MLForge v2.0  |  Machine Learning Development Platform  |  ")
                    fr.font.color.rgb = RGBColor(0x66,0x66,0x80); fr.font.size=Pt(8)
                    
                    # Page numbers in Word using field codes
                    fldChar1 = OxmlElement('w:fldChar')
                    fldChar1.set(qn('w:fldCharType'), 'begin')
                    instrText = OxmlElement('w:instrText')
                    instrText.set(qn('xml:space'), 'preserve')
                    instrText.text = "PAGE"
                    fldChar2 = OxmlElement('w:fldChar')
                    fldChar2.set(qn('w:fldCharType'), 'end')
                    
                    r_element = fp.add_run()._r
                    r_element.append(fldChar1)
                    r_element.append(instrText)
                    r_element.append(fldChar2)
                    
                    fp.runs[-1].font.color.rgb = RGBColor(0x66,0x66,0x80)
                    fp.runs[-1].font.size=Pt(8)

                wb2 = io.BytesIO(); dw.save(wb2); wb2.seek(0)
                st.download_button("MLForge_Report.docx", data=wb2.getvalue(),
                    file_name="MLForge_Report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                st.success("Word document ready.")
            except Exception as e:
                st.error(f"Word error: {e}")
                with st.expander("Debug"): st.code(traceback.format_exc())

    # ═════ RAW DATA & MODEL DOWNLOADS ═════════════════════════════════
    st.markdown("---")
    sh("", "Data & Model Export", "Download your intermediate datasets and trained models in various formats")
    st.markdown("<br>", unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    
    for col, fmt_badge, fmt_title, fmt_desc, fmt_icon in [
        (dc1, "CLEAN",        "Cleaned Dataset",      "Data after missing values & duplicates handled",  hgi("missing_values",  size=40, color="accent")),
        (dc2, "PREPROCESSED", "Preprocessed Dataset", "Fully transformed, encoded & scaled features",   hgi("preprocessing",   size=40, color="accent")),
        (dc3, "MODEL",        "Trained Model",        "Best trained pipeline architecture & weights",   hgi("model",           size=40, color="accent")),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card" style="min-height:{CARD_H};margin-bottom:1rem">
            <div style="font-size:2.5rem;margin-bottom:.3rem">{fmt_icon}</div>
            <div style="font-weight:700;color:var(--accent2);font-size:1.05rem">{fmt_title}</div>
            <div style="font-size:.76rem;color:var(--muted);margin-top:.4rem;line-height:1.5">{fmt_desc}</div>
            </div>""", unsafe_allow_html=True)

    # Dropdowns row
    sc1, sc2, sc3 = st.columns(3)
    opts_data = ["CSV (.csv)", "Excel (.xlsx)", "JSON (.json)", "XML (.xml)", "YAML (.yaml)", "SQLite DB (.db)"]
    with sc1: c_fmt = st.selectbox("Format", opts_data, key="fmt_clean", label_visibility="collapsed")
    with sc2: p_fmt = st.selectbox("Format", opts_data, key="fmt_prep", label_visibility="collapsed")
    with sc3: m_fmt = st.selectbox("Format", ["Pickle (.pkl)", "ONNX (.onnx)", "Joblib (.joblib)"], key="fmt_model", label_visibility="collapsed")

    def get_dl_data(df, fmt):
        if "CSV" in fmt: return df.to_csv(index=False).encode('utf-8'), "text/csv", "csv"
        elif "Excel" in fmt:
            buf = io.BytesIO(); df.to_excel(buf, index=False); return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
        elif "JSON" in fmt: return df.to_json(orient='records').encode('utf-8'), "application/json", "json"
        elif "XML" in fmt:
            import re
            df_xml = df.copy()
            df_xml.columns = [re.sub(r'\W', '_', str(c)) if not str(c)[:1].isdigit() else '_' + re.sub(r'\W', '_', str(c)) for c in df.columns]
            return df_xml.to_xml(index=False).encode('utf-8'), "application/xml", "xml"
        elif "YAML" in fmt:
            import yaml; return yaml.dump(df.to_dict(orient='records'), sort_keys=False).encode('utf-8'), "application/x-yaml", "yaml"
        else: # SQLite
            import sqlite3, tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp: tmp_name = tmp.name
            conn = sqlite3.connect(tmp_name); df.to_sql("dataset", conn, index=False, if_exists="replace"); conn.close()
            with open(tmp_name, "rb") as f: data_bytes = f.read()
            os.remove(tmp_name)
            return data_bytes, "application/octet-stream", "db"

    # Buttons row
    bc1, bc2, bc3 = st.columns(3)
    
    with bc1:
        df_c = st.session_state.get("df_cleaned_only", st.session_state.get("df_raw"))
        if df_c is not None:
            data_bytes, mime, ext = get_dl_data(df_c, c_fmt)
            st.download_button(f"Download {ext.upper()}", data=data_bytes, file_name=f"clean_dataset.{ext}", mime=mime, use_container_width=True)
        else:
            st.button("No Clean Data", disabled=True, use_container_width=True, key="btn_c_dis")
            
    with bc2:
        df_p = st.session_state.get("df_clean")
        if df_p is not None:
            data_bytes, mime, ext = get_dl_data(df_p, p_fmt)
            st.download_button(f"Download {ext.upper()}", data=data_bytes, file_name=f"preprocessed_dataset.{ext}", mime=mime, use_container_width=True)
        else:
            st.button("No Preprocessed Data", disabled=True, use_container_width=True, key="btn_p_dis")
            
    with bc3:
        best_name = st.session_state.get("best_model_name")
        trained_models = st.session_state.get("trained_models", {})
        if best_name and best_name in trained_models:
            model_obj = trained_models[best_name]
            f_type = m_fmt.split()[0].lower()
            X_sample = st.session_state.get("X_test")
            if X_sample is None: X_sample = st.session_state.get("df_clean")
            if X_sample is not None: X_sample = X_sample.head(1)
            model_bytes, mime_or_err = export_model(model_obj, f_type, X_sample)
            if model_bytes:
                ext = "zip" if f_type == "tensorflow" else ("pt" if f_type == "torch" else m_fmt.split(" (.")[1][:-1])
                st.download_button(f"Download {ext.upper()}", data=model_bytes, file_name=f"{best_name.replace(' ','_')}.{ext}", mime=mime_or_err, use_container_width=True)
            else:
                st.button(f"Error: {mime_or_err[:15]}...", disabled=True, use_container_width=True, help=mime_or_err, key="btn_m_err")
        else:
            st.button("No Trained Model", disabled=True, use_container_width=True, key="btn_m_dis")

    #  New Pipeline 
    st.markdown("---")
    st.markdown("""<div class="ai-card" style="text-align:center">
    <strong>Start a New Pipeline</strong><br>
    <span style="color:var(--muted);font-size:.85rem">Reset session state and begin fresh with a new dataset</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    def _on_start_new_pipeline():
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    render_step_navigation(
        back_step=8,
        back_label="◀ Back to Visualizer",
        next_label="Start New Pipeline ▶",
        on_next_click=_on_start_new_pipeline
    )
    footer()


# STEP 10 — PREDICTION PLAYGROUND

elif st.session_state.step == 10:
    sh("", "Prediction Playground & Explanations", "Interactive inference and local prediction attribution")
    render_prediction_playground_ui(
        trained_models=st.session_state.get("trained_models", {}),
        default_model_name=st.session_state.get("best_model_name"),
        problem_type=st.session_state.get("problem_type", "classification"),
        feature_schema=st.session_state.get("feature_schema"),
        X_background=st.session_state.get("X_train"),
        target_col=st.session_state.get("target_col")
    )

    st.markdown("<br>", unsafe_allow_html=True)
    render_step_navigation(
        back_step=7,
        back_label="◀ Back to Model Training",
        next_step=9,
        next_label="Next → Report & Export ▶"
    )
    footer()


# STEP 11 — EXPLAINABILITY (SHAP & Model Interpretability)

elif st.session_state.step == 11:
    best_name = st.session_state.get("best_model_name")
    trained_models = st.session_state.get("trained_models", {})
    best_obj = trained_models.get(best_name) if best_name else None
    problem_type = st.session_state.get("problem_type", "classification")

    if not best_obj:
        render_empty_state(
            title="No Trained Model Available for Explainability",
            description="Train a machine learning model in Step 7 (Model Training) first to inspect global feature importance, SHAP attribution values, and local prediction explanations.",
            cta_label="Go to Model Training",
            cta_step=7,
            icon="brain"
        )
        st.stop()

    sh("", "Model Explainability & Interpretability", "Inspect global feature importances, TreeSHAP/LinearSHAP attribution, and local prediction decisions")
    render_model_explanation_ui(
        pipeline_or_model=best_obj,
        model_name=best_name,
        problem_type=problem_type,
        X_sample=st.session_state.get("X_test"),
        y_sample=st.session_state.get("y_test"),
        X_background=st.session_state.get("X_train")
    )

    st.markdown("<br>", unsafe_allow_html=True)
    render_step_navigation(
        back_step=7,
        back_label="◀ Back to Model Training",
        back_key="btn_exp_back",
        next_step=10,
        next_label="Next → Prediction Playground ▶",
        next_key="btn_exp_next"
    )
    footer()


