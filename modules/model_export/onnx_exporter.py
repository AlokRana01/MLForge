def export_onnx(model, X_sample):
    try:
        if X_sample is None:
            return None

        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
        import numpy as np
        import pandas as pd

        # Handle Pipeline with preprocessor step
        if hasattr(model, "named_steps"):
            # Attempt direct conversion with schema if DataFrame
            if isinstance(X_sample, pd.DataFrame):
                try:
                    initial_types = []
                    for col in X_sample.columns:
                        if pd.api.types.is_numeric_dtype(X_sample[col]):
                            initial_types.append((str(col), FloatTensorType([None, 1])))
                        else:
                            initial_types.append((str(col), StringTensorType([None, 1])))
                    onnx_model = convert_sklearn(model, initial_types=initial_types)
                    return onnx_model.SerializeToString()
                except Exception:
                    pass

            # Fallback for pipelines: convert the estimator step using transformed sample
            if "preprocessor" in model.named_steps and "model" in model.named_steps:
                try:
                    preprocessor = model.named_steps["preprocessor"]
                    estimator = model.named_steps["model"]
                    X_trans = preprocessor.transform(X_sample)
                    X_trans = np.asarray(X_trans, dtype=np.float32)
                    initial_type = [('input', FloatTensorType([None, X_trans.shape[1]]))]
                    onnx_model = convert_sklearn(estimator, initial_types=initial_type)
                    return onnx_model.SerializeToString()
                except Exception:
                    pass

        # Standalone model or numpy input
        if hasattr(X_sample, "values"):
            X_sample = X_sample.values
        X_sample = np.asarray(X_sample, dtype=np.float32)

        initial_type = [('input', FloatTensorType([None, X_sample.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        return onnx_model.SerializeToString()

    except Exception as e:
        return None