"""
Comprehensive Test Suite for Leakage-Safe Preprocessing & Cross-Validation
===========================================================================
Covers all 16 required test scenarios:
1. Numerical imputation leakage
2. Categorical encoding leakage
3. Scaling leakage
4. Train/test split independence
5. Stratified classification split
6. Cross-validation execution & metrics (mean, std)
7. Small dataset handling
8. Imbalanced classification with class weighting
9. End-to-end regression
10. End-to-end classification
11. Missing values handled inside pipeline
12. Unseen categorical values during prediction (no crash)
13. Constant columns in features
14. Feature mismatch detection
15. Target accidentally included in features check
16. Deliberate leakage scenario with extreme test outliers
"""

import io
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from modules.automl import (
    build_preprocessor,
    build_model_pipeline,
    prepare_train_test_split,
    run_cross_validation,
    train_and_evaluate_models,
    validate_training_data,
    identify_feature_columns
)
from modules.evaluation import (
    evaluate_classification,
    evaluate_regression,
    build_results_table,
    is_higher_better
)


# =====================================================================
# 1. Numerical imputation leakage test
# =====================================================================
def test_numerical_imputation_leakage():
    # Train set has values 10, 20, 30 (median = 20)
    # Test set has values 1000, 2000, 3000 (median = 2000)
    X_train = pd.DataFrame({"num": [10.0, 20.0, 30.0, np.nan]})
    y_train = pd.Series([0, 1, 0, 1])
    X_test = pd.DataFrame({"num": [1000.0, 2000.0, 3000.0, np.nan]})

    pipe = build_model_pipeline(LogisticRegression(), numeric_cols=["num"], categorical_cols=[])
    pipe.fit(X_train, y_train)

    # Inspect the learned imputer parameter inside the pipeline
    imputer = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    learned_median = imputer.statistics_[0]

    # Must equal median of X_train non-nulls (20.0), completely unaffected by X_test
    assert learned_median == 20.0
    assert learned_median != 2000.0


# =====================================================================
# 2. Categorical encoding leakage test
# =====================================================================
def test_categorical_encoding_leakage():
    # Train set only contains "A" and "B"
    # Test set contains "A", "B", and unseen "Z"
    X_train = pd.DataFrame({"cat": ["A", "B", "A", "B"]})
    y_train = pd.Series([0, 1, 0, 1])
    X_test = pd.DataFrame({"cat": ["A", "B", "Z"]})

    pipe = build_model_pipeline(LogisticRegression(), numeric_cols=[], categorical_cols=["cat"])
    pipe.fit(X_train, y_train)

    # Inspect the learned encoder categories inside the pipeline
    encoder = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    learned_cats = list(encoder.categories_[0])

    # Must only contain train categories
    assert "A" in learned_cats
    assert "B" in learned_cats
    assert "Z" not in learned_cats

    # Predicting on X_test containing unseen "Z" must NOT raise an error
    preds = pipe.predict(X_test)
    assert len(preds) == 3


# =====================================================================
# 3. Scaling leakage test
# =====================================================================
def test_scaling_leakage():
    # Train set values: [10, 20, 30] (mean = 20, std = 8.165)
    # Test set values: [1000, 2000, 3000]
    X_train = pd.DataFrame({"feat": [10.0, 20.0, 30.0]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"feat": [1000.0, 2000.0, 3000.0]})

    pipe = build_model_pipeline(LinearRegression(), numeric_cols=["feat"], categorical_cols=[])
    pipe.fit(X_train, y_train)

    scaler = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    learned_mean = scaler.mean_[0]

    # Learned mean must be exactly 20.0 from X_train
    assert np.isclose(learned_mean, 20.0)


# =====================================================================
# 4. Train/test split independence
# =====================================================================
def test_train_test_split_independence():
    df = pd.DataFrame({
        "x1": range(100),
        "x2": np.random.randn(100),
        "target": np.random.choice([0, 1], size=100)
    })
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.25, random_state=42
    )

    # Check sizes
    assert len(X_train) == 75
    assert len(X_test) == 25
    # Strict index independence
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0
    # Target isolated
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns


# =====================================================================
# 5. Stratified classification split
# =====================================================================
def test_stratified_classification_split():
    # 80 zeros and 20 ones (20% positive class ratio)
    y_raw = [0] * 80 + [1] * 20
    df = pd.DataFrame({
        "feat": np.random.randn(100),
        "label": y_raw
    })
    X_train, X_test, y_train, y_test, _, _ = prepare_train_test_split(
        df, target_col="label", test_size=0.20, random_state=42, problem_type="classification"
    )

    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Stratification ensures positive class ratio is approximately preserved (20%)
    assert np.isclose(train_ratio, 0.20, atol=0.03)
    assert np.isclose(test_ratio, 0.20, atol=0.03)


# =====================================================================
# 6. Cross-validation execution & metrics (mean, std)
# =====================================================================
def test_cross_validation_execution():
    X_train = pd.DataFrame({
        "num": np.random.randn(60),
        "cat": np.random.choice(["A", "B", "C"], size=60)
    })
    y_train = pd.Series(np.random.choice([0, 1], size=60))

    pipe = build_model_pipeline(
        LogisticRegression(max_iter=1000),
        numeric_cols=["num"],
        categorical_cols=["cat"]
    )

    fold_scores, cv_mean, cv_std, splits = run_cross_validation(
        pipe, X_train, y_train,
        problem_type="classification",
        n_splits=5,
        optimization_metric="Accuracy"
    )

    assert splits == 5
    assert len(fold_scores) == 5
    assert 0.0 <= cv_mean <= 1.0
    assert 0.0 <= cv_std <= 1.0
    assert np.isclose(cv_mean, np.mean(fold_scores), atol=1e-4)


# =====================================================================
# 7. Small dataset handling
# =====================================================================
def test_small_dataset_handling():
    # 8 samples
    X_small = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    y_small = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    pipe = build_model_pipeline(LogisticRegression(), numeric_cols=["f"], categorical_cols=[])

    # Request 5 folds on 8 samples; engine should automatically reduce splits without crashing
    fold_scores, cv_mean, cv_std, splits = run_cross_validation(
        pipe, X_small, y_small,
        problem_type="classification",
        n_splits=5,
        optimization_metric="Accuracy"
    )
    assert splits <= 4
    assert len(fold_scores) == splits


# =====================================================================
# 8. Imbalanced classification with class weighting
# =====================================================================
def test_imbalanced_classification_weighting():
    # 95 class 0, 5 class 1 (severe imbalance)
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3, n_redundant=1,
        weights=[0.95, 0.05], random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
    df["target"] = y

    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.20, random_state=42, problem_type="classification"
    )

    results_df, trained_models, best_model = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        problem_type="classification",
        selected_models=["LogisticRegression", "RandomForest"],
        optimization_metric="Balanced Accuracy",
        is_imbalanced=True,
        cv_folds=3
    )

    assert not results_df.empty
    assert "Balanced Accuracy" in results_df.columns
    assert "CV Mean" in results_df.columns
    assert "CV Std" in results_df.columns
    # Verify class_weight is balanced
    rf_pipeline = trained_models["RandomForest"]
    assert rf_pipeline.named_steps["model"].class_weight == "balanced"


# =====================================================================
# 9. End-to-end regression
# =====================================================================
def test_end_to_end_regression():
    X, y = make_regression(n_samples=80, n_features=3, noise=0.1, random_state=42)  # type: ignore
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target_val"] = y

    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target_val", test_size=0.20, random_state=42, problem_type="regression"
    )

    results_df, trained_models, best_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        problem_type="regression",
        selected_models=["LinearRegression", "RandomForest"],
        optimization_metric="R2 Score",
        cv_folds=3
    )

    assert not results_df.empty
    assert "R2 Score" in results_df.columns
    assert "RMSE" in results_df.columns
    assert "MAE" in results_df.columns
    assert best_name in trained_models


# =====================================================================
# 10. End-to-end classification
# =====================================================================
def test_end_to_end_classification():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.20, random_state=42, problem_type="classification"
    )

    results_df, trained_models, best_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        problem_type="classification",
        selected_models=["LogisticRegression", "RandomForest", "DecisionTree"],
        optimization_metric="F1 Score",
        cv_folds=3
    )

    assert len(results_df) == 3
    assert "CV Mean" in results_df.columns
    assert "F1 Score" in results_df.columns
    assert "Accuracy" in results_df.columns


# =====================================================================
# 11. Missing values handled natively inside pipeline
# =====================================================================
def test_missing_values_handled_inside_pipeline():
    # Dataset with missing values in both numeric and categorical features
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0] * 5,
        "cat": ["A", "B", np.nan, "A", "B", np.nan, "A", "B"] * 5,
        "y": [0, 1, 0, 1, 0, 1, 0, 1] * 5
    })

    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="y", test_size=0.25, random_state=42, problem_type="classification"
    )

    # Note: X_train and X_test still contain NaNs! The pipeline must impute them.
    assert X_train["num"].isna().sum() > 0
    assert X_train["cat"].isna().sum() > 0

    results_df, trained_models, best_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        problem_type="classification",
        selected_models=["LogisticRegression"],
        cv_folds=3
    )

    assert not results_df.empty
    # Raw prediction on test set with NaNs must succeed
    pipe = trained_models["LogisticRegression"]
    preds = pipe.predict(X_test)
    assert len(preds) == len(X_test)


# =====================================================================
# 12. Unseen categorical values during prediction (no crash)
# =====================================================================
def test_unseen_categorical_values_during_prediction():
    X_train = pd.DataFrame({"city": ["London", "Paris", "Berlin"] * 10})
    y_train = pd.Series([0, 1, 0] * 10)
    # Test set has completely novel categories "Tokyo" and "Sydney"
    X_test_unseen = pd.DataFrame({"city": ["Tokyo", "Sydney", "Paris"]})

    pipe = build_model_pipeline(LogisticRegression(), numeric_cols=[], categorical_cols=["city"])
    pipe.fit(X_train, y_train)

    # Should predict without error (OneHotEncoder ignore unknown categories)
    preds = pipe.predict(X_test_unseen)
    assert len(preds) == 3


# =====================================================================
# 13. Constant columns in features
# =====================================================================
def test_constant_columns_in_features():
    df = pd.DataFrame({
        "const_feat": [42.0] * 40,
        "normal_feat": np.random.randn(40),
        "target": np.random.choice([0, 1], size=40)
    })

    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_train_test_split(
        df, target_col="target", test_size=0.20, random_state=42
    )

    pipe = build_model_pipeline(LogisticRegression(), numeric_cols=num_cols, categorical_cols=cat_cols)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    assert len(preds) == len(X_test)


# =====================================================================
# 14. Feature mismatch detection
# =====================================================================
def test_feature_mismatch_detection():
    X_bad = pd.DataFrame({"a": [1, 2]})
    y_bad = pd.Series([1, 2, 3])  # Length 3 != 2
    with pytest.raises(ValueError, match="Length mismatch"):
        validate_training_data(X_bad, y_bad)


# =====================================================================
# 15. Target accidentally included in features check
# =====================================================================
def test_target_accidentally_included_in_features():
    X = pd.DataFrame({"feat": [1, 2, 3], "target_col": [0, 1, 0]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(ValueError, match="Target Protection Alert"):
        validate_training_data(X, y, target_col="target_col")


# =====================================================================
# 16. Deliberate leakage scenario: Extreme outlier in test set
# =====================================================================
def test_deliberate_leakage_outlier_test():
    """
    Deliberately inject an extreme outlier into the test set (1,000,000.0).
    Verify that the pipeline scaler's learned mean and scale match X_train ONLY,
    and are completely untainted by the test set outlier.
    """
    # 50 samples in train with mean = 10.0
    train_vals = np.array([10.0] * 50)
    # 10 samples in test with an extreme outlier of 1,000,000.0
    test_vals = np.array([10.0] * 9 + [1_000_000.0])

    X_train = pd.DataFrame({"signal": train_vals})
    y_train = pd.Series([0, 1] * 25)
    X_test = pd.DataFrame({"signal": test_vals})
    y_test = pd.Series([0, 1] * 5)

    pipe = build_model_pipeline(LinearRegression(), numeric_cols=["signal"], categorical_cols=[])
    # Train pipeline strictly on X_train
    pipe.fit(X_train, y_train)

    # Extract learned scaler parameters
    scaler = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    learned_mean = float(scaler.mean_[0])

    # If leakage occurred (e.g. scaling was done before split on X_full),
    # the learned mean would be ~16,675.0 instead of 10.0!
    assert np.isclose(learned_mean, 10.0, atol=1e-5), f"Data Leakage Detected! Scaler mean was {learned_mean}, expected 10.0."
    assert not np.isclose(learned_mean, np.mean(np.concatenate([train_vals, test_vals])))


# =====================================================================
# 17. Pipeline export and reloadability
# =====================================================================
def test_pipeline_export_and_reload():
    from modules.model_export.export_manager import export_model
    import pickle
    import joblib

    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cat": ["a", "b", "a", "b", "a", "b"],
        "target": [0, 1, 0, 1, 0, 1]
    })
    pipe = build_model_pipeline(LogisticRegression(), numeric_cols=["num"], categorical_cols=["cat"])
    pipe.fit(df[["num", "cat"]], df["target"])

    # Test joblib serialization and reloading
    data_j, mime_j = export_model(pipe, "joblib")
    assert data_j is not None
    assert isinstance(data_j, bytes)
    reloaded_j = joblib.load(io.BytesIO(data_j))
    preds_j = reloaded_j.predict(df[["num", "cat"]])
    assert len(preds_j) == 6

    # Test pickle serialization and reloading
    data_p, mime_p = export_model(pipe, "pickle")
    assert data_p is not None
    assert isinstance(data_p, bytes)
    reloaded_p = pickle.loads(data_p)
    preds_p = reloaded_p.predict(df[["num", "cat"]])
    assert len(preds_p) == 6

    # Test ONNX export
    data_o, mime_o = export_model(pipe, "onnx", df[["num", "cat"]].head(2))
    assert data_o is not None
    assert isinstance(data_o, bytes)

