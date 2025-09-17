import pickle
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model():
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro")
    }

    # Save metrics report
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("📊 Metrics Report:", metrics)

    # Fail pipeline if accuracy < 0.8
    if metrics["accuracy"] < 0.8:
        raise ValueError("❌ Model accuracy below threshold!")

if __name__ == "__main__":
    evaluate_model()
