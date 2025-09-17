from src.train import train_model
from src.predict import predict

def test_prediction_is_correct():
    train_model()
    result = predict([6, 2.5, 4.8, 1.8])  # pass 4 features
    assert result in [0, 1, 2]
