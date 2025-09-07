import pytest
from mnist import Net
from mnist import main
@pytest.fixture
def model():
    return Net()

def test_total_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params}')
    assert total_params <= 25_000, "Model should have less than 25000 parameters."

def test_accuracy():
    train_accuracy, test_accuracy = main()
    assert test_accuracy >= 95.0, f"Test accuracy is {test_accuracy}%, should be at least 95%"