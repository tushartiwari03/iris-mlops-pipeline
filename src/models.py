from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_model(model_type):
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=200)
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100)
    elif model_type == "svm":
        return SVC()
    else:
        raise ValueError("Unsupported model type")
