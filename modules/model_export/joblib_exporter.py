import joblib
import tempfile
import os

def export_joblib(model):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
    temp.close()

    joblib.dump(model, temp.name)

    with open(temp.name, "rb") as f:
        data = f.read()

    os.remove(temp.name)
    return data