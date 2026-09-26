def export_torch(model):
    try:
        import torch
        import tempfile
        import os

        if not hasattr(model, "state_dict"):
            return None

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        temp.close()

        try:
            torch.save(model.state_dict(), temp.name)

            with open(temp.name, "rb") as f:
                data = f.read()

        finally:
            os.remove(temp.name)

        return data

    except Exception as e:
        return None