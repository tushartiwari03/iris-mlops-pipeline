import yaml
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models import get_model

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Load data
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"], random_state=42
)

# Get model
model = get_model(config["model_type"])
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"Model: {config['model_type']}")
print(f"Accuracy: {accuracy}")

# Quality gate
if accuracy < config["min_accuracy"]:
    raise Exception("Accuracy below threshold")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/iris_model.joblib")
