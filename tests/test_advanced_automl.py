"""
Comprehensive Test Suite for Advanced AutoML + Hyperparameter Optimization
===========================================================================
Covers all 13 mandatory test criteria for Loop Engineering Task 3:
1. Model registry coverage & instantiation (9 classification, 10 regression)
2. Stage A baseline screening
3. Stage B hyperparameter optimization with RandomizedSearchCV
4. Cross-validation ranking
5. Metric selection directionality (higher_is_better vs lower_is_better)
6. Failed model isolation (single model failure does not crash the run)
7. Reproducibility verification (identical random_state -> identical results)
8. Small dataset handling (N < 20)
9. Binary classification pipeline
10. Multiclass classification pipeline
11. Regression pipeline
12. Imbalanced classification with class weighting
13. Best-model selection with deterministic explanation justification
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.base import BaseEstimator, ClassifierMixin

from modules.automl import (
    get_default_models,
    get_hyperparameter_search_spaces,
    prepare_train_test_split,
    train_and_evaluate_models,
    AutoMLResult
)
from modules.evaluation import is_higher_better, get_sklearn_scoring


# =====================================================================
# 1. Model Registry
# =====================================================================
def test_model_registry_coverage():
    clf_models = get_default_models(problem_type="classification", random_state=42)
    reg_models = get_default_models(problem_type="regression", random_state=42)

    # 9 Classifiers
    expected_clfs = {
        "LogisticRegression", "DecisionTree", "RandomForest", "ExtraTrees",
        "GradientBoosting", "HistGradientBoosting", "SVC", "KNN", "NaiveBayes"
    }
    assert expected_clfs.issubset(set(clf_models.keys())), f"Missing classifiers: {expected_clfs - set(clf_models.keys())}"

    # 10 Regressors
    expected_regs = {
        "LinearRegression", "Ridge", "Lasso", "ElasticNet",
        "DecisionTree", "RandomForest", "ExtraTrees", "GradientBoosting",
        "HistGradientBoosting", "SVR"
    }
    assert expected_regs.issubset(set(reg_models.keys())), f"Missing regressors: {expected_regs - set(reg_models.keys())}"

    # Verify search spaces exist for primary models
    for name in ["LogisticRegression", "DecisionTree", "RandomForest", "ExtraTrees", "Ridge", "Lasso"]:
        space = get_hyperparameter_search_spaces(name)
        assert len(space) > 0, f"Search space empty for {name}"
        assert all(k.startswith("model__") for k in space.keys())


# =====================================================================
# 2. Stage A: Baseline Screening
# =====================================================================
def test_stage_a_baseline_screening():
    iris = load_iris(as_frame=True)
    df = iris.frame
    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.2, random_state=42
    )

    result = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["LogisticRegression", "DecisionTree"],
        optimization_metric="Accuracy",
        cv_folds=3,
        enable_tuning=False,
        random_state=42
    )

    assert isinstance(result, AutoMLResult)
    df_res, trained_ms, best_name = result

    assert len(df_res) == 2
    assert "Train Time (s)" in df_res.columns
    assert "Stage" in df_res.columns
    assert all(df_res["Stage"] == "Baseline")
    assert all(df_res["Train Time (s)"] >= 0)
    assert best_name in ["LogisticRegression", "DecisionTree"]


# =====================================================================
# 3. Stage B: Hyperparameter Optimization
# =====================================================================
def test_stage_b_hyperparameter_optimization():
    iris = load_iris(as_frame=True)
    df = iris.frame
    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.2, random_state=42
    )

    result = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["DecisionTree"],
        optimization_metric="F1 Score",
        cv_folds=3,
        enable_tuning=True,
        tune_top_k=1,
        tune_n_iter=3,
        random_state=42
    )

    df_res, trained_ms, best_name = result
    # Should include both Baseline and Tuned
    models_evaluated = df_res["Model"].tolist()
    assert "DecisionTree" in models_evaluated
    assert "DecisionTree (Tuned)" in models_evaluated

    tuned_row = df_res[df_res["Model"] == "DecisionTree (Tuned)"].iloc[0]
    assert tuned_row["Stage"] == "Tuned"
    assert tuned_row["Optimization Status"] == "Tuned (RandomizedSearchCV)"
    assert isinstance(tuned_row["Best Hyperparameters"], dict)
    assert len(tuned_row["Best Hyperparameters"]) > 0


# =====================================================================
# 4. Cross-Validation Ranking
# =====================================================================
def test_cv_ranking_accuracy():
    X, y = make_classification(n_samples=80, n_features=6, n_informative=3, n_redundant=1, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y

    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42
    )

    result = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["DecisionTree", "RandomForest", "NaiveBayes"],
        optimization_metric="Balanced Accuracy",
        cv_folds=3,
        enable_tuning=False
    )

    df_res = result.results_df
    # Verify descending sort for Balanced Accuracy
    scores = df_res["Balanced Accuracy"].tolist()
    assert scores == sorted(scores, reverse=True)


# =====================================================================
# 5. Metric Selection Directionality (Higher vs Lower is Better)
# =====================================================================
def test_metric_directionality_sorting():
    X, y = make_regression(n_samples=80, n_features=4, noise=5.0, random_state=42)  # type: ignore
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="regression"
    )

    # MAE is lower is better
    result_mae = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="regression",
        selected_models=["LinearRegression", "Ridge", "DecisionTree"],
        optimization_metric="MAE",
        cv_folds=3,
        enable_tuning=False
    )
    df_mae = result_mae.results_df
    mae_scores = df_mae["MAE"].tolist()
    assert mae_scores == sorted(mae_scores, reverse=False), "MAE must be sorted ascending (lower is better)"

    # R2 is higher is better
    result_r2 = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="regression",
        selected_models=["LinearRegression", "Ridge", "DecisionTree"],
        optimization_metric="R2 Score",
        cv_folds=3,
        enable_tuning=False
    )
    df_r2 = result_r2.results_df
    r2_scores = df_r2["R2 Score"].tolist()
    assert r2_scores == sorted(r2_scores, reverse=True), "R2 must be sorted descending (higher is better)"


# =====================================================================
# 6. Failed Model Isolation (Fault Tolerance)
# =====================================================================
class BrokenEstimator(BaseEstimator, ClassifierMixin):
    """Simulates a buggy third-party estimator that throws during fit."""
    def fit(self, X, y):
        raise RuntimeError("Simulated internal estimator crash.")
    def predict(self, X):
        return np.zeros(len(X))


def test_failed_model_isolation():
    iris = load_iris(as_frame=True)
    df = iris.frame
    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.2, random_state=42
    )

    # Inject broken model alongside valid DecisionTree
    from modules.automl import get_default_models
    import modules.automl as automl_mod

    orig_get_default = automl_mod.get_default_models

    def mock_get_default(problem_type="classification", is_imbalanced=False, random_state=42):
        models = orig_get_default(problem_type, is_imbalanced, random_state)
        models["BrokenModel"] = BrokenEstimator()
        return models

    automl_mod.get_default_models = mock_get_default
    try:
        result = train_and_evaluate_models(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
            selected_models=["DecisionTree", "BrokenModel"],
            optimization_metric="Accuracy",
            cv_folds=2
        )

        df_res, trained_ms, best_name = result
        # AutoML must NOT crash; DecisionTree must succeed
        assert best_name == "DecisionTree"
        assert "DecisionTree" in df_res["Model"].tolist()
        assert "BrokenModel" not in df_res["Model"].tolist()

        # Check that error was isolated and recorded in provenance
        failed_list = result.provenance.get("failed_models", [])
        assert any(f["model"] == "BrokenModel" for f in failed_list)
    finally:
        automl_mod.get_default_models = orig_get_default


# =====================================================================
# 7. Reproducibility
# =====================================================================
def test_reproducibility():
    X, y = make_classification(n_samples=60, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42
    )

    # Run 1
    res1 = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["RandomForest"],
        cv_folds=3,
        enable_tuning=True,
        tune_n_iter=3,
        random_state=123
    )

    # Run 2 with identical seed
    res2 = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["RandomForest"],
        cv_folds=3,
        enable_tuning=True,
        tune_n_iter=3,
        random_state=123
    )

    assert res1.best_model_name == res2.best_model_name
    np.testing.assert_allclose(
        res1.results_df["CV Mean"].values,
        res2.results_df["CV Mean"].values,
        atol=1e-5
    )


# =====================================================================
# 8. Small Dataset Handling (N < 20)
# =====================================================================
def test_small_dataset_handling():
    df_small = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "b": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df_small, target_col="target", test_size=0.25, random_state=42
    )

    result = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["LogisticRegression", "DecisionTree"],
        cv_folds=5,  # Requested 5 folds on small dataset
        enable_tuning=True,
        tune_top_k=1,
        tune_n_iter=2,
        random_state=42
    )
    assert not result.results_df.empty
    assert result.best_model_name is not None


# =====================================================================
# 9. Binary Classification Pipeline
# =====================================================================
def test_binary_classification_pipeline():
    X, y = make_classification(n_samples=60, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42
    )

    result = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["HistGradientBoosting", "ExtraTrees"],
        optimization_metric="ROC AUC",
        cv_folds=3,
        random_state=42
    )
    assert len(result.results_df) == 2
    assert "ROC AUC" in result.results_df.columns


# =====================================================================
# 10. Multiclass Classification Pipeline
# =====================================================================
def test_multiclass_classification_pipeline():
    X, y = make_classification(
        n_samples=75, n_classes=3, n_informative=4, n_clusters_per_class=1, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42
    )

    result = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["DecisionTree", "RandomForest"],
        optimization_metric="F1 Score",
        cv_folds=3,
        random_state=42
    )
    assert len(result.results_df) == 2
    assert result.best_model_name is not None


# =====================================================================
# 11. Regression Pipeline
# =====================================================================
def test_regression_pipeline():
    X, y = make_regression(n_samples=70, n_features=5, noise=0.1, random_state=42)  # type: ignore
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.2, random_state=42, problem_type="regression"
    )

    result = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="regression",
        selected_models=["Lasso", "ElasticNet", "HistGradientBoosting"],
        optimization_metric="RMSE",
        cv_folds=3,
        enable_tuning=False,
        random_state=42
    )
    assert len(result.results_df) == 3
    assert "RMSE" in result.results_df.columns
    assert result.results_df["RMSE"].min() >= 0.0


# =====================================================================
# 12. Imbalanced Classification with Class Weighting
# =====================================================================
def test_imbalanced_classification_weighting():
    # 92 vs 8 samples
    X, y = make_classification(n_samples=100, weights=[0.92, 0.08], random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y

    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.2, random_state=42
    )

    result = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["LogisticRegression", "ExtraTrees"],
        optimization_metric="Balanced Accuracy",
        is_imbalanced=True,
        cv_folds=2,
        random_state=42
    )
    # Check that class_weight was balanced in fitted estimator
    for m_name in ["LogisticRegression", "ExtraTrees"]:
        estimator = result.trained_models[m_name].named_steps["model"]
        assert estimator.class_weight == "balanced"


# =====================================================================
# 13. Best-Model Selection with Deterministic Justification
# =====================================================================
def test_best_model_selection_and_justification():
    iris = load_iris(as_frame=True)
    df = iris.frame
    X_train, X_test, y_train, y_test, num_c, cat_c = prepare_train_test_split(
        df, target_col="target", test_size=0.2, random_state=42
    )

    result = train_and_evaluate_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        numeric_cols=num_c, categorical_cols=cat_c, problem_type="classification",
        selected_models=["DecisionTree", "RandomForest"],
        optimization_metric="Accuracy",
        cv_folds=3,
        random_state=42
    )

    best_name = result.best_model_name
    assert best_name in ["DecisionTree", "RandomForest"]

    # Verify deterministic explanation string
    reason = result.selection_reason
    assert f"Selected '{best_name}' based on Accuracy" in reason
    assert "cross-validation Accuracy" in reason
    assert "holdout test score" in reason
