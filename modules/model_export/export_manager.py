from .pickle_exporter import export_pickle
from .joblib_exporter import export_joblib
from .onnx_exporter import export_onnx
from .tensorflow_exporter import export_tensorflow
from .torch_exporter import export_torch


def export_model(model, format_type, X_sample=None):
    format_type = format_type.lower()

    try:
        if format_type == "pickle":
            return export_pickle(model), "application/octet-stream"

        elif format_type == "joblib":
            return export_joblib(model), "application/octet-stream"

        elif format_type == "onnx":
            if X_sample is None:
                return None, "X_sample required for ONNX export"
            data = export_onnx(model, X_sample)
            if data is None:
                return None, "ONNX conversion is unsupported for this pipeline or model architecture. Please use Joblib or Pickle."
            return data, "application/octet-stream"

        elif format_type == "tensorflow":
            data = export_tensorflow(model)
            return data, "application/zip"

        elif format_type == "torch":
            data = export_torch(model)
            return data, "application/octet-stream"

        else:
            return None, "Unsupported format"

    except Exception as e:
        return None, f"Error: {str(e)}"