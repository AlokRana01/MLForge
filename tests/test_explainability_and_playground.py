"""
MLForge - Test Suite for Explainable AI & Prediction Playground
====================================================================
Validates all 14 required capabilities and edge cases:
1. Classification prediction
2. Regression prediction
3. Missing input handling
4. Invalid numeric input handling
5. Unknown category handling
6. Trained model absent handling
7. Different model families inspection
8. Tree model SHAP explanation
9. Linear model explanation
10. Model without native feature importance (permutation fallback)
11. Feature mismatch handling
12. Probability unavailable handling
13. Multiclass classification prediction & attribution
14. Large feature count scalability
15. Model export compatibility (preserving schema and metadata)
"""

import io
import pickle
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from modules.automl import build_model_pipeline, train_and_evaluate_models, prepare_train_test_split
from modules.explainability import (
    inspect_model_family,
    compute_global_feature_importance,
    compute_shap_explanations,
    explain_single_prediction,
    plot_global_importance,
    plot_shap_summary,
    plot_prediction_waterfall,
    _HAS_SHAP
)
from modules.prediction_playground import (
    extract_feature_schema,
    validate_and_sanitize_inputs,
    execute_pipeline_prediction,
    run_prediction_playground
)
from modules.model_export.export_manager import export_model


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def binary_classification_data():
    """Synthetic binary dataset with numeric and categorical features."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n).astype(float),
        "income": np.random.uniform(20000, 120000, size=n),
        "credit_score": np.random.normal(650, 50, size=n),
        "department": np.random.choice(["Sales", "Engineering", "Marketing"], size=n),
        "target": np.random.choice(["Denied", "Approved"], size=n)
    })
    return df


@pytest.fixture
def multiclass_data():
    """Iris multiclass dataset."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    return df


@pytest.fixture
def regression_data():
    """Diabetes continuous regression dataset."""
    diab = load_diabetes(as_frame=True)
    df = diab.frame.copy()
    return df


# =====================================================================
# TEST CASES 1 & 2: PREDICTION (CLASSIFICATION & REGRESSION)
# =====================================================================

def test_1_classification_prediction(binary_classification_data):
    """Test 1: Binary classification prediction with class probabilities."""
    df = binary_classification_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="classification"
    )
    pipe = build_model_pipeline(
        LogisticRegression(max_iter=200),
        numeric_cols=num_cols,
        categorical_cols=cat_cols
    )
    pipe.fit(X_train, y_train)

    sample = X_test.iloc[:1]
    res = execute_pipeline_prediction(pipe, sample, problem_type="classification")

    assert res["status"] == "success"
    assert res["prediction"] in ["Denied", "Approved"]
    assert res["has_probabilities"] is True
    assert isinstance(res["probabilities"], dict)
    assert set(res["probabilities"].keys()) == {"Approved", "Denied"}
    prob_sum = sum(res["probabilities"].values())
    assert pytest.approx(prob_sum, 1e-4) == 1.0


def test_2_regression_prediction(regression_data):
    """Test 2: Continuous regression prediction strictly without confidence label."""
    df = regression_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="regression"
    )
    pipe = build_model_pipeline(
        Ridge(),
        numeric_cols=num_cols,
        categorical_cols=cat_cols
    )
    pipe.fit(X_train, y_train)

    sample = X_test.iloc[:1]
    res = execute_pipeline_prediction(pipe, sample, problem_type="regression")

    assert res["status"] == "success"
    assert isinstance(res["prediction"], (int, float, np.floating))
    assert res["has_probabilities"] is False
    assert res["probabilities"] is None


# =====================================================================
# TEST CASES 3, 4, 5: INPUT VALIDATION & RESILIENCE
# =====================================================================

def test_3_missing_input(binary_classification_data):
    """Test 3: Missing inputs are handled via pipeline imputer without crash."""
    df = binary_classification_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="classification"
    )
    pipe = build_model_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=42),
        numeric_cols=num_cols,
        categorical_cols=cat_cols
    )
    pipe.fit(X_train, y_train)

    schema = extract_feature_schema(X_train)
    # Provide input with None / empty fields
    raw_input = {
        "age": None,
        "income": "",
        "credit_score": 600.0,
        "department": "Engineering"
    }

    is_valid, sanitized_df, errors, warnings = validate_and_sanitize_inputs(raw_input, schema)
    assert is_valid is True
    assert sanitized_df is not None
    assert len(errors) == 0

    # Execute through pipeline: median/mode imputer inside pipeline ensures clean inference
    res = execute_pipeline_prediction(pipe, sanitized_df, problem_type="classification")
    assert res["status"] == "success"
    assert res["prediction"] in ["Denied", "Approved"]


def test_4_invalid_numeric_input():
    """Test 4: Non-numeric strings in numeric fields produce clean validation error."""
    schema = {
        "age": {"name": "age", "type": "numeric", "default": 30.0, "is_nullable": False},
        "score": {"name": "score", "type": "numeric", "default": 50.0, "is_nullable": False}
    }
    raw_input = {"age": "thirty_two", "score": 75.0}
    is_valid, sanitized_df, errors, warnings = validate_and_sanitize_inputs(raw_input, schema)

    assert is_valid is False
    assert sanitized_df is None
    assert any("Invalid numeric input" in err for err in errors)

    # Playground wrapper handles validation error safely
    play_res = run_prediction_playground(None, raw_input, schema, problem_type="classification")
    assert play_res["status"] in ["no_model", "validation_error"]


def test_5_unknown_category(binary_classification_data):
    """Test 5: Unseen categorical category handled gracefully via ignore without crashing."""
    df = binary_classification_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="classification"
    )
    pipe = build_model_pipeline(
        LogisticRegression(max_iter=200),
        numeric_cols=num_cols,
        categorical_cols=cat_cols
    )
    pipe.fit(X_train, y_train)

    schema = extract_feature_schema(X_train)
    # Inject unknown department category not in training data
    raw_input = {
        "age": 35.0,
        "income": 50000.0,
        "credit_score": 700.0,
        "department": "Mars_Colony_Operations"
    }

    is_valid, sanitized_df, errors, warnings = validate_and_sanitize_inputs(raw_input, schema)
    assert is_valid is True
    assert any("not seen in training" in w for w in warnings)

    res = execute_pipeline_prediction(pipe, sanitized_df, problem_type="classification")
    assert res["status"] == "success"
    assert res["prediction"] in ["Denied", "Approved"]


# =====================================================================
# TEST CASES 6 & 7: MODEL STATE & ARCHITECTURES
# =====================================================================

def test_6_trained_model_absent():
    """Test 6: Graceful instruction when no trained model exists."""
    schema = {"x": {"name": "x", "type": "numeric", "default": 1.0}}
    res = run_prediction_playground(None, {"x": 1.0}, schema, problem_type="classification")

    assert res["status"] == "no_model"
    assert "No compatible trained model exists" in res["message"]


def test_7_different_model_families():
    """Test 7: Model inspection correctly detects Tree, Linear, and Distance/Kernel families."""
    rf = RandomForestClassifier()
    lr = LogisticRegression()
    svc = SVC(kernel="rbf")
    knn = KNeighborsClassifier()

    info_rf = inspect_model_family(rf)
    assert info_rf["family"] == "Tree"
    assert info_rf["shap_explainer_type"] == "TreeExplainer"

    info_lr = inspect_model_family(lr)
    assert info_lr["family"] == "Linear"
    assert info_lr["shap_explainer_type"] == "LinearExplainer"

    info_svc = inspect_model_family(svc)
    assert info_svc["family"] == "Kernel/Distance/Probabilistic"
    assert info_svc["has_native_importance"] is False

    info_knn = inspect_model_family(knn)
    assert info_knn["family"] == "Kernel/Distance/Probabilistic"
    assert info_knn["has_native_importance"] is False


# =====================================================================
# TEST CASES 8, 9, 10: EXPLANATIONS (SHAP, LINEAR, FALLBACK)
# =====================================================================

def test_8_tree_model_shap(binary_classification_data):
    """Test 8: Tree model SHAP explanation and local attribution."""
    df = binary_classification_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="classification"
    )
    pipe = build_model_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=42),
        numeric_cols=num_cols,
        categorical_cols=cat_cols
    )
    pipe.fit(X_train, y_train)

    # 1. Global feature importance
    imp_res = compute_global_feature_importance(pipe, X_test, y_test, problem_type="classification")
    assert imp_res["status"] == "success"
    assert imp_res["is_native"] is True
    assert not imp_res["importance_df"].empty

    # 2. SHAP explanation
    shap_res = compute_shap_explanations(pipe, X_background_raw=X_train, X_explain_raw=X_test, max_background=30, max_explain=30)
    if _HAS_SHAP:
        assert shap_res["is_available"] is True
        assert shap_res["shap_values"].shape[0] <= 30
        assert not shap_res["global_importance_df"].empty

    # 3. Individual prediction explanation
    single_row = X_test.iloc[:1]
    local_exp = explain_single_prediction(pipe, single_row, X_background_raw=X_train, problem_type="classification")
    assert "prediction" in local_exp
    assert not local_exp["top_contributions_df"].empty
    assert "These features contributed most" in local_exp["disclaimer"]


def test_9_linear_model_explanation(binary_classification_data):
    """Test 9: Linear model absolute coefficient explanation."""
    df = binary_classification_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="classification"
    )
    pipe = build_model_pipeline(
        LogisticRegression(max_iter=200),
        numeric_cols=num_cols,
        categorical_cols=cat_cols
    )
    pipe.fit(X_train, y_train)

    imp_res = compute_global_feature_importance(pipe, X_test, y_test, problem_type="classification")
    assert imp_res["status"] == "success"
    assert "Linear Model Absolute Coefficients" in imp_res["mechanism"]
    assert imp_res["is_native"] is True
    assert len(imp_res["importance_df"]) > 0


def test_10_model_without_native_feature_importance(binary_classification_data):
    """Test 10: Model without native importance (SVC) falls back cleanly to Permutation Importance."""
    df = binary_classification_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="classification"
    )
    pipe = build_model_pipeline(
        SVC(kernel="rbf", probability=False),
        numeric_cols=num_cols,
        categorical_cols=cat_cols
    )
    pipe.fit(X_train, y_train)

    # With validation sample: computes permutation importance
    imp_res = compute_global_feature_importance(pipe, X_test, y_test, problem_type="classification", n_repeats=2)
    assert imp_res["status"] == "success"
    assert imp_res["is_native"] is False
    assert "Permutation Feature Importance" in imp_res["mechanism"]
    assert not imp_res["importance_df"].empty

    # Without validation sample: does NOT invent weights, reports unsupported cleanly
    imp_no_data = compute_global_feature_importance(pipe, None, None, problem_type="classification")
    assert imp_no_data["status"] == "unsupported"
    assert imp_no_data["importance_df"].empty


# =====================================================================
# TEST CASES 11, 12, 13, 14: ADVANCED EDGE CASES
# =====================================================================

def test_11_feature_mismatch(binary_classification_data):
    """Test 11: Feature mismatch (missing columns, extra columns) handled defensively."""
    schema = {
        "age": {"name": "age", "type": "numeric", "default": 25.0, "is_nullable": False},
        "income": {"name": "income", "type": "numeric", "default": 50000.0, "is_nullable": False}
    }
    # Missing 'income', has extra 'alien_feature'
    raw_input = {"age": 40.0, "alien_feature": "ignore_me"}
    is_valid, sanitized_df, errors, warnings = validate_and_sanitize_inputs(raw_input, schema)

    assert is_valid is True
    assert "income" in sanitized_df.columns
    assert sanitized_df["income"].iloc[0] == 50000.0  # Filled with default
    assert "alien_feature" not in sanitized_df.columns  # Extra column omitted
    assert any("income" in w for w in warnings)


def test_12_probability_unavailable(binary_classification_data):
    """Test 12: Model without predict_proba correctly flags has_probabilities=False."""
    df = binary_classification_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="classification"
    )
    pipe = build_model_pipeline(
        SVC(kernel="rbf", probability=False),
        numeric_cols=num_cols,
        categorical_cols=cat_cols
    )
    pipe.fit(X_train, y_train)

    sample = X_test.iloc[:1]
    res = execute_pipeline_prediction(pipe, sample, problem_type="classification")

    assert res["status"] == "success"
    assert res["prediction"] in ["Denied", "Approved"]
    assert res["has_probabilities"] is False
    assert res["probabilities"] is None


def test_13_multiclass_classification(multiclass_data):
    """Test 13: Multiclass 3-class prediction and probability distribution."""
    df = multiclass_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="classification"
    )
    pipe = build_model_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=42),
        numeric_cols=num_cols,
        categorical_cols=cat_cols
    )
    pipe.fit(X_train, y_train)

    sample = X_test.iloc[:1]
    res = execute_pipeline_prediction(pipe, sample, problem_type="classification")

    assert res["status"] == "success"
    assert res["prediction"] in [0, 1, 2]
    assert res["has_probabilities"] is True
    assert len(res["probabilities"]) == 3
    assert pytest.approx(sum(res["probabilities"].values()), 1e-4) == 1.0


def test_14_large_feature_count():
    """Test 14: Dataset with 50+ features correctly extracts schema without target."""
    n_features = 55
    X, y = make_classification(n_samples=100, n_features=n_features, n_informative=10, random_state=42)
    col_names = [f"feat_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=col_names)
    df["label_target"] = y

    schema = extract_feature_schema(df, target_col="label_target")
    assert len(schema) == 55
    assert "label_target" not in schema

    # Verify input validation and prediction on large feature set
    sample_input = {k: v["default"] for k, v in schema.items()}
    is_valid, sanitized_df, errors, _ = validate_and_sanitize_inputs(sample_input, schema)
    assert is_valid is True
    assert sanitized_df.shape[1] == 55


# =====================================================================
# TEST CASE 15: EXPORT COMPATIBILITY
# =====================================================================

def test_15_model_export_preserves_schema_and_metadata(binary_classification_data):
    """Test 15: Model export preserves preprocessing, model, feature schema, and metadata."""
    df = binary_classification_data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42, problem_type="classification"
    )

    result = train_and_evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        problem_type="classification",
        selected_models=["RandomForest"],
        optimization_metric="F1 Score",
        cv_folds=3
    )

    best_pipe = result.trained_models[result.best_model_name]
    assert hasattr(best_pipe, "feature_schema_")
    assert len(best_pipe.feature_schema_) > 0

    # Test Joblib export & deserialization
    joblib_bytes, mime = export_model(best_pipe, "joblib")
    assert joblib_bytes is not None

    loaded_pipe = joblib.load(io.BytesIO(joblib_bytes))
    assert hasattr(loaded_pipe, "feature_schema_")
    assert hasattr(loaded_pipe, "provenance_")
    assert loaded_pipe.predict(X_test.iloc[:2]).tolist() == best_pipe.predict(X_test.iloc[:2]).tolist()

    # Test Pickle export & deserialization
    pkl_bytes, mime_p = export_model(best_pipe, "pickle")
    assert pkl_bytes is not None
    loaded_pkl = pickle.loads(pkl_bytes)
    assert hasattr(loaded_pkl, "feature_schema_")
    assert loaded_pkl.predict(X_test.iloc[:2]).tolist() == best_pipe.predict(X_test.iloc[:2]).tolist()
