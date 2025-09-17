import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(values):
    values = np.array([values])  # values should be list of 4 features
    return model.predict(values)[0]
