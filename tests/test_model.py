import os

def test_model_file_exists():
    assert os.path.exists("models/iris_model.joblib")
