# test_model.py

from train import train_model

def test_model_accuracy():
    acc = train_model()
    assert acc > 0.9, f"Accuracy is too low: {acc}"

if __name__ == "__main__":
    test_model_accuracy()
