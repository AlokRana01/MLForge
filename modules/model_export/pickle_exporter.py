import pickle
import tempfile
import os

def export_pickle(model):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    temp.close()

    with open(temp.name, "wb") as f:
        pickle.dump(model, f)

    with open(temp.name, "rb") as f:
        data = f.read()

    os.remove(temp.name)
    return data