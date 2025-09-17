from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

def train_model():
    X, y = load_iris(return_X_y=True)  # Iris dataset (4 features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as model.pkl")

if __name__ == "__main__":
    train_model()
