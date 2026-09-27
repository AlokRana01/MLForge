"""
MLForge - Leakage-Safe AutoML & Cross-Validation Engine
=============================================================
Provides statistically reliable machine learning model training:
- Strict leakage prevention: all parameter learning (imputation, scaling, encoding)
  is encapsulated inside an sklearn Pipeline with ColumnTransformer.
- Proper train/test splitting: stratified for classification, random for regression.
- Configurable K-Fold / StratifiedKFold cross-validation with automatic fold reduction.
- Reporting of both Cross-Validation performance (CV Mean, CV Std) and Holdout Test performance.
- User-selectable optimization metric for leaderboard ranking and best model selection.
- Automatic class-weight balancing for severe class imbalance.
- Full prediction compatibility: trained pipeline directly accepts raw feature inputs.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV

# Evaluation utilities
from modules.evaluation import (
    evaluate_model, evaluate_classification, evaluate_regression,
    build_results_table, is_higher_better, get_available_metrics,
    get_sklearn_scoring
)


# =====================================================================
# PROBLEM TYPE DETECTION
# =====================================================================

def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Robustly determine whether the task is classification or regression.
    """
    if target_col not in df.columns:
        return "classification"

    y = df[target_col].dropna()
    if y.empty:
        return "classification"

    if y.dtype == "object" or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"

    if pd.api.types.is_numeric_dtype(y):
        unique_threshold = max(10, int(0.05 * len(y)))
        if y.nunique() < unique_threshold:
            return "classification"
        return "regression"

    return "classification"


# =====================================================================
# COLUMN IDENTIFICATION
# =====================================================================

def identify_feature_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical feature column names from a DataFrame.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # Categoricals include object, category, boolean, string
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


# =====================================================================
# LEAKAGE-SAFE PREPROCESSOR & PIPELINE BUILDER
# =====================================================================

def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler_type: str = "standard"
) -> ColumnTransformer:
    """
    Build an sklearn ColumnTransformer ensuring that:
    1. Imputation and scaling for numeric features are fitted strictly on train data.
    2. Imputation and one-hot encoding for categorical features are fitted strictly on train data.
    3. Unseen categorical values during test/prediction are ignored (no crashes).
    """
    transformers = []

    # Numeric sub-pipeline
    if numeric_cols:
        scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler)
        ])
        transformers.append(("num", num_pipeline, numeric_cols))

    # Categorical sub-pipeline
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("Cannot build preprocessor: both numeric and categorical feature lists are empty.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    model: Any,
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler_type: str = "standard"
) -> Pipeline:
    """
    Encapsulate preprocessor and model inside an end-to-end Pipeline.
    The returned pipeline accepts raw DataFrame feature inputs directly.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scaler_type=scaler_type)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# =====================================================================
# DEFENSIVE INPUT VALIDATION
# =====================================================================

def validate_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    target_col: Optional[str] = None
) -> None:
    """
    Perform defensive integrity checks to prevent data leakage and shape mismatches.
    """
    if X is None or X.empty:
        raise ValueError("Feature matrix X cannot be empty.")
    if y is None or len(y) == 0:
        raise ValueError("Target vector y cannot be empty.")
    if len(X) != len(y):
        raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y)} rows.")

    # Check 1: Target column cannot appear as a feature
    if target_col is not None and target_col in X.columns:
        raise ValueError(f"Target Protection Alert: Target column '{target_col}' was found inside feature matrix X.")

    # Check 2: Check for direct target duplicates in X
    for col in X.columns:
        try:
            if X[col].equals(y):
                raise ValueError(f"Leakage Protection Alert: Feature column '{col}' is an exact copy of target.")
        except Exception:
            pass


# =====================================================================
# LEAKAGE-SAFE TRAIN / TEST SPLIT
# =====================================================================

def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.20,
    random_state: int = 42,
    problem_type: str = "classification"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Partition dataset into training and holdout test sets without leakage.
    - Strips target column completely from features.
    - Drops rows where target is NaN (labels required for training).
    - Uses stratified split for classification when valid (all classes >= 2 samples).
    - Returns X_train, X_test, y_train, y_test, numeric_cols, categorical_cols.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")

    # Drop rows where target is missing
    valid_df = df.dropna(subset=[target_col]).copy()
    if valid_df.empty:
        raise ValueError(f"Target column '{target_col}' contains only NaN values. No supervised samples available.")

    X = valid_df.drop(columns=[target_col])
    y = valid_df[target_col]

    validate_training_data(X, y, target_col=target_col)

    numeric_cols, categorical_cols = identify_feature_columns(X)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No feature columns remaining after separating target.")

    n_samples = len(y)
    stratify = None

    if problem_type == "classification":
        class_counts = y.value_counts()
        n_classes = len(class_counts)
        min_class = class_counts.min()

        # Stratification is valid only if every class has at least 2 samples
        # and test set can contain at least 1 sample per class
        if min_class >= 2 and (n_samples * test_size) >= n_classes and n_samples >= (n_classes * 2):
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Defensive check: ensure train and test indices are completely disjoint
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0, "Leakage Alert: Train and test sets share index rows."

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


# =====================================================================
# CROSS-VALIDATION EVALUATION
# =====================================================================

def run_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str = "classification",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    optimization_metric: str = "F1 Score"
) -> Tuple[List[float], float, float, int]:
    """
    Run leakage-safe K-Fold cross-validation on the training set:
    - Preprocessing is fitted strictly inside each fold.
    - Automatically reduces fold count if dataset size or minority class count is small.
    - Evaluates the requested optimization metric across folds.
    - Returns (fold_scores, cv_mean, cv_std, actual_splits_used).
    """
    n_samples = len(y_train)
    if n_samples < 4:
        # Dataset too small for multi-fold CV
        return [0.0], 0.0, 0.0, 1

    # Adapt fold count
    actual_splits = max(2, min(n_splits, n_samples // 2))

    if problem_type == "classification":
        class_counts = y_train.value_counts()
        min_count = int(class_counts.min()) if not class_counts.empty else 0
        if min_count >= 2:
            actual_splits = max(2, min(actual_splits, min_count))
            cv = StratifiedKFold(n_splits=actual_splits, shuffle=shuffle, random_state=random_state)
        else:
            cv = KFold(n_splits=actual_splits, shuffle=shuffle, random_state=random_state)
    else:
        cv = KFold(n_splits=actual_splits, shuffle=shuffle, random_state=random_state)

    fold_scores = []
    classes = np.unique(y_train) if problem_type == "classification" else None

    for train_idx, val_idx in cv.split(X_train, y_train if problem_type == "classification" else None):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Re-clone pipeline so each fold starts completely fresh
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_tr, y_tr)

        y_pred = fold_pipeline.predict(X_val)
        y_proba = None
        if problem_type == "classification" and hasattr(fold_pipeline, "predict_proba"):
            try:
                y_proba = fold_pipeline.predict_proba(X_val)
            except Exception:
                y_proba = None

        metrics = evaluate_model(y_val, y_pred, problem_type=problem_type, y_proba=y_proba, classes=classes)
        score = metrics.get(optimization_metric)
        if score is None or np.isnan(score):
            # Fallback to secondary metric if primary metric is undefined
            fallback = "Accuracy" if problem_type == "classification" else "R2 Score"
            score = metrics.get(fallback, 0.0)

        fold_scores.append(float(score) if score is not None else 0.0)

    cv_mean = round(float(np.mean(fold_scores)), 4)
    cv_std = round(float(np.std(fold_scores)), 4)

    return fold_scores, cv_mean, cv_std, actual_splits


# =====================================================================
# MODEL FACTORY
# =====================================================================

def get_default_models(
    problem_type: str = "classification",
    is_imbalanced: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Get dictionary of base estimators with balanced class weighting if imbalanced.
    Expands model library to include HistGradientBoosting, ExtraTrees, Lasso, and ElasticNet.
    """
    cw = "balanced" if is_imbalanced else None

    if problem_type == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import (
            RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
            HistGradientBoostingClassifier
        )
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB

        return {
            "LogisticRegression": LogisticRegression(max_iter=2000, random_state=random_state, class_weight=cw),
            "DecisionTree": DecisionTreeClassifier(random_state=random_state, class_weight=cw),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight=cw),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=random_state, class_weight=cw),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=random_state),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state, class_weight=cw),
            "SVC": SVC(probability=True, random_state=random_state, class_weight=cw),
            "KNN": KNeighborsClassifier(),
            "NaiveBayes": GaussianNB()
        }
    else:
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import (
            RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
            HistGradientBoostingRegressor
        )
        from sklearn.svm import SVR

        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=random_state),
            "Lasso": Lasso(random_state=random_state, max_iter=2000),
            "ElasticNet": ElasticNet(random_state=random_state, max_iter=2000),
            "DecisionTree": DecisionTreeRegressor(random_state=random_state),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=random_state),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=random_state),
            "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
            "SVR": SVR(kernel="rbf")
        }


# =====================================================================
# HYPERPARAMETER SEARCH SPACES
# =====================================================================

def get_hyperparameter_search_spaces(
    model_name: str,
    problem_type: str = "classification",
    n_train_samples: int = 100
) -> Dict[str, Any]:
    """
    Return bounded hyperparameter search space for RandomizedSearchCV.
    All parameters are scoped with the 'model__' pipeline prefix.
    """
    clean_name = model_name.replace(" (Tuned)", "").strip()

    # Dynamic safe neighbor range for KNN
    max_k = max(1, int(n_train_samples * 0.6))
    valid_k = [k for k in [3, 5, 7, 9] if k <= max_k] or [1]

    spaces: Dict[str, Dict[str, Any]] = {
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"]
        },
        "DecisionTree": {
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "RandomForest": {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "ExtraTrees": {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "GradientBoosting": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 5]
        },
        "HistGradientBoosting": {
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_iter": [50, 100, 150],
            "model__max_leaf_nodes": [15, 31, 63]
        },
        "SVC": {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto", 0.01, 0.1]
        },
        "SVR": {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto", 0.01, 0.1]
        },
        "KNN": {
            "model__n_neighbors": valid_k,
            "model__weights": ["uniform", "distance"]
        },
        "NaiveBayes": {
            "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        },
        "LinearRegression": {
            "model__fit_intercept": [True, False]
        },
        "Ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0]
        },
        "Lasso": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        "ElasticNet": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.7, 0.9]
        }
    }
    return spaces.get(clean_name, {})


# =====================================================================
# AUTOML RESULT CONTAINER
# =====================================================================

class AutoMLResult:
    """
    Container for AutoML execution results.
    Unpacks as a 3-tuple (results_df, trained_models, best_model_name)
    for 100% backwards compatibility, while exposing provenance metadata
    and deterministic selection justification.
    """
    def __init__(
        self,
        results_df: pd.DataFrame,
        trained_models: Dict[str, Pipeline],
        best_model_name: Optional[str],
        provenance: Optional[Dict[str, Any]] = None,
        selection_reason: Optional[str] = None,
        feature_schema: Optional[Dict[str, Any]] = None,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None
    ):
        self.results_df = results_df
        self.trained_models = trained_models
        self.best_model_name = best_model_name
        self.provenance = provenance or {}
        self.selection_reason = selection_reason or ""
        self.feature_schema = feature_schema or {}
        self.X_train = X_train
        self.y_train = y_train

    def __iter__(self):
        return iter((self.results_df, self.trained_models, self.best_model_name))

    def __getitem__(self, idx):
        return (self.results_df, self.trained_models, self.best_model_name)[idx]

    def __len__(self):
        return 3

    def __repr__(self):
        return f"<AutoMLResult: best_model='{self.best_model_name}', models_evaluated={len(self.results_df)}>"


# =====================================================================
# FULL ADVANCED AUTOML TRAINING PIPELINE (TWO-STAGE)
# =====================================================================

def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    problem_type: str = "classification",
    selected_models: Optional[List[str]] = None,
    optimization_metric: str = "F1 Score",
    cv_folds: int = 5,
    scaler_type: str = "standard",
    is_imbalanced: bool = False,
    enable_tuning: bool = False,
    tune_top_k: int = 2,
    tune_n_iter: int = 10,
    random_state: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> AutoMLResult:
    """
    Execute full leakage-safe Two-Stage AutoML:
    - Stage A: Baseline screening across selected model candidates with CV and holdout testing.
    - Stage B: Hyperparameter optimization on top K candidates using RandomizedSearchCV.
    - Fault-tolerant model isolation: individual model errors do not crash the batch.
    - Deterministic model selection based on user-chosen optimization metric.
    - Reproducibility provenance metadata tracking.

    Returns:
    - AutoMLResult (unpacks as results_df, trained_models, best_model_name).
    """
    all_models = get_default_models(problem_type=problem_type, is_imbalanced=is_imbalanced, random_state=random_state)
    if selected_models:
        model_pool = {k: v for k, v in all_models.items() if k in selected_models}
    else:
        model_pool = all_models

    if not model_pool:
        raise ValueError("No valid models selected for training.")

    results: List[Dict[str, Any]] = []
    trained_pipelines: Dict[str, Pipeline] = {}
    failed_models: List[Dict[str, str]] = []
    classes = np.unique(np.concatenate([y_train, y_test])) if problem_type == "classification" else None

    total_stage_a = len(model_pool)
    tune_candidates_count = min(tune_top_k, total_stage_a) if enable_tuning else 0
    total_steps = total_stage_a + tune_candidates_count
    current_step = 0

    # -----------------------------------------------------------------
    # STAGE A: BASELINE SCREENING
    # -----------------------------------------------------------------
    for name, base_model in model_pool.items():
        current_step += 1
        if progress_callback:
            progress_callback(
                current_step / total_steps,
                f"Stage A: Screening {name} ({cv_folds}-fold CV)..."
            )

        t0 = time.perf_counter()
        try:
            # 1. Build pipeline template
            pipe = build_model_pipeline(
                base_model,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                scaler_type=scaler_type
            )

            # 2. Run Cross-Validation on training set
            fold_scores, cv_mean, cv_std, actual_folds = run_cross_validation(
                pipe,
                X_train,
                y_train,
                problem_type=problem_type,
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state,
                optimization_metric=optimization_metric
            )

            # 3. Final fit of complete pipeline on full training data
            pipe.fit(X_train, y_train)
            trained_pipelines[name] = pipe

            # 4. Holdout evaluation on test set
            y_pred = pipe.predict(X_test)
            y_proba = None
            if problem_type == "classification" and hasattr(pipe, "predict_proba"):
                try:
                    y_proba = pipe.predict_proba(X_test)
                except Exception:
                    y_proba = None

            test_metrics = evaluate_model(
                y_test, y_pred, problem_type=problem_type, y_proba=y_proba, classes=classes
            )
            elapsed_sec = round(time.perf_counter() - t0, 3)

            # 5. Structured record for baseline
            record: Dict[str, Any] = {
                "Model": name,
                "Stage": "Baseline",
                "Optimization Status": "Baseline",
                "CV Mean": cv_mean,
                "CV Std": cv_std,
                "Test Score": test_metrics.get(optimization_metric, 0.0),
                "Train Time (s)": elapsed_sec,
                "Best Hyperparameters": "Default"
            }
            for m_name, m_val in test_metrics.items():
                record[m_name] = m_val

            results.append(record)

        except Exception as e:
            # Isolate failure: record and continue evaluating remaining models
            failed_models.append({"model": name, "stage": "Baseline", "error": str(e)})
            print(f"[WARNING] Baseline screening failed for {name}: {e}")

    # -----------------------------------------------------------------
    # STAGE B: HYPERPARAMETER OPTIMIZATION (ON TOP CANDIDATES)
    # -----------------------------------------------------------------
    if enable_tuning and len(results) > 0 and len(X_train) >= 10:
        # Rank baseline results by optimization metric
        higher_better = is_higher_better(optimization_metric)
        sorted_baselines = sorted(
            results,
            key=lambda r: r.get(optimization_metric, r.get("CV Mean", 0.0)),
            reverse=higher_better
        )

        # Pick top K distinct model architectures that have search spaces
        candidates_to_tune: List[str] = []
        for r in sorted_baselines:
            m_name = r["Model"]
            if m_name in model_pool and get_hyperparameter_search_spaces(m_name, problem_type, len(X_train)):
                candidates_to_tune.append(m_name)
            if len(candidates_to_tune) >= tune_top_k:
                break

        for tune_name in candidates_to_tune:
            current_step += 1
            tuned_model_key = f"{tune_name} (Tuned)"
            if progress_callback:
                progress_callback(
                    current_step / total_steps,
                    f"Stage B: Optimizing hyperparameters for {tune_name}..."
                )

            t0 = time.perf_counter()
            try:
                base_estimator = clone(model_pool[tune_name])
                pipe = build_model_pipeline(
                    base_estimator,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    scaler_type=scaler_type
                )

                search_space = get_hyperparameter_search_spaces(tune_name, problem_type, len(X_train))
                scoring_str = get_sklearn_scoring(optimization_metric, problem_type)

                # Set up cross-validation splitter clamped to sample size
                actual_cv = min(cv_folds, len(X_train))
                if problem_type == "classification":
                    min_class_samples = y_train.value_counts().min()
                    actual_cv = max(2, min(actual_cv, min_class_samples))
                    cv_splitter = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=random_state)
                else:
                    cv_splitter = KFold(n_splits=max(2, actual_cv), shuffle=True, random_state=random_state)

                # Safe iteration count
                n_iter_safe = max(2, min(tune_n_iter, 20))

                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=search_space,
                    n_iter=n_iter_safe,
                    cv=cv_splitter,
                    scoring=scoring_str,
                    random_state=random_state,
                    n_jobs=1,
                    error_score="raise"
                )
                search.fit(X_train, y_train)

                best_pipe = search.best_estimator_
                raw_best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
                trained_pipelines[tuned_model_key] = best_pipe

                # Evaluate tuned pipeline with standard cross-validation
                _, cv_mean, cv_std, _ = run_cross_validation(
                    best_pipe,
                    X_train,
                    y_train,
                    problem_type=problem_type,
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=random_state,
                    optimization_metric=optimization_metric
                )

                # Holdout test set evaluation
                y_pred = best_pipe.predict(X_test)
                y_proba = None
                if problem_type == "classification" and hasattr(best_pipe, "predict_proba"):
                    try:
                        y_proba = best_pipe.predict_proba(X_test)
                    except Exception:
                        y_proba = None

                test_metrics = evaluate_model(
                    y_test, y_pred, problem_type=problem_type, y_proba=y_proba, classes=classes
                )
                elapsed_sec = round(time.perf_counter() - t0, 3)

                record = {
                    "Model": tuned_model_key,
                    "Stage": "Tuned",
                    "Optimization Status": "Tuned (RandomizedSearchCV)",
                    "CV Mean": cv_mean,
                    "CV Std": cv_std,
                    "Test Score": test_metrics.get(optimization_metric, 0.0),
                    "Train Time (s)": elapsed_sec,
                    "Best Hyperparameters": raw_best_params
                }
                for m_name, m_val in test_metrics.items():
                    record[m_name] = m_val

                results.append(record)

            except Exception as e:
                failed_models.append({"model": tuned_model_key, "stage": "Tuned", "error": str(e)})
                print(f"[WARNING] Tuning failed for {tune_name}: {e}")

    if progress_callback:
        progress_callback(1.0, "AutoML execution finished!")

    # -----------------------------------------------------------------
    # RANKING & BEST MODEL SELECTION
    # -----------------------------------------------------------------
    results_df = build_results_table(
        results,
        sort_metric=optimization_metric,
        problem_type=problem_type
    )

    best_model_name = results_df.iloc[0]["Model"] if not results_df.empty else None

    # Deterministic selection justification
    selection_reason = ""
    best_params_extracted: Dict[str, Any] = {}
    if not results_df.empty:
        best_row = results_df.iloc[0]
        direction_phrase = "highest" if is_higher_better(optimization_metric) else "lowest"
        best_cv_m = best_row.get("CV Mean", 0.0)
        best_cv_s = best_row.get("CV Std", 0.0)
        best_t_s = best_row.get("Test Score", 0.0)
        best_params_extracted = best_row.get("Best Hyperparameters", {})

        selection_reason = (
            f"Selected '{best_model_name}' based on {optimization_metric}: "
            f"it achieved the {direction_phrase} cross-validation {optimization_metric} "
            f"({best_cv_m:.4f} +/- {best_cv_s:.4f}) and holdout test score ({best_t_s:.4f}) "
            f"across all {len(results_df)} evaluated configurations."
        )

    # Provenance metadata for reproducibility
    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "best_model_name": best_model_name,
        "optimization_metric": optimization_metric,
        "optimization_direction": "maximize" if is_higher_better(optimization_metric) else "minimize",
        "cv_folds": cv_folds,
        "tuning_enabled": enable_tuning,
        "tuning_top_k": tune_top_k if enable_tuning else 0,
        "tuning_iterations": tune_n_iter if enable_tuning else 0,
        "dataset_dimensions": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_count": X_train.shape[1]
        },
        "feature_columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols
        },
        "preprocessing_config": {
            "scaler": scaler_type,
            "imputation_numeric": "median",
            "imputation_categorical": "most_frequent",
            "encoding": "onehot_ignore_unknown"
        },
        "best_hyperparameters": best_params_extracted,
        "failed_models": failed_models,
        "selection_reason": selection_reason
    }

    # Attach feature schema and metadata directly to pipeline objects for export portability
    from modules.prediction_playground import extract_feature_schema
    feature_schema = extract_feature_schema(X_train)

    for pipe_obj in trained_pipelines.values():
        try:
            pipe_obj.feature_schema_ = feature_schema
            pipe_obj.problem_type_ = problem_type
            pipe_obj.provenance_ = provenance
        except Exception:
            pass

    return AutoMLResult(
        results_df=results_df,
        trained_models=trained_pipelines,
        best_model_name=best_model_name,
        provenance=provenance,
        selection_reason=selection_reason,
        feature_schema=feature_schema,
        X_train=X_train,
        y_train=y_train
    )


# =====================================================================
# BACKWARDS COMPATIBILITY WRAPPERS
# =====================================================================

def prepare_data(df: pd.DataFrame, target_col: str):
    """
    Backwards-compatible data preparation returning (X_train, X_test, y_train, y_test).
    Uses leakage-safe splitting without mutating raw features.
    """
    prob_type = detect_problem_type(df, target_col)
    X_train, X_test, y_train, y_test, _, _ = prepare_train_test_split(
        df, target_col=target_col, test_size=0.2, random_state=42, problem_type=prob_type
    )
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test, problem_type):
    """
    Backwards-compatible entry point for model training.
    """
    num_cols, cat_cols = identify_feature_columns(X_train)
    opt_metric = "F1 Score" if problem_type == "classification" else "R2 Score"
    results_df, trained_ms, _ = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        problem_type=problem_type,
        optimization_metric=opt_metric,
        cv_folds=5
    )
    return results_df, trained_ms


def get_best_model(results_df: pd.DataFrame, problem_type: str) -> Optional[pd.Series]:
    """
    Backwards-compatible best model row extractor.
    """
    if results_df.empty:
        return None
    return results_df.iloc[0]