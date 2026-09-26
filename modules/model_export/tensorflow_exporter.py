def export_tensorflow(model):
    try:
        import tempfile
        import shutil
        import os

        temp_dir = tempfile.mkdtemp()
        model.save(temp_dir)

        zip_path = temp_dir + ".zip"
        shutil.make_archive(temp_dir, 'zip', temp_dir)

        with open(zip_path, "rb") as f:
            data = f.read()

        shutil.rmtree(temp_dir)
        os.remove(zip_path)

        return data

    except Exception as e:
        return None