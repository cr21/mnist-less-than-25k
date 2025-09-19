import pytest
from mnist import Net
from mnist import main
import torch.nn as nn

@pytest.fixture
def model():
    return Net()

def test_total_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params}')
    assert total_params <= 25_000, "Model should have less than 25000 parameters."

def test_batch_normalization_exists(model):
    has_bn = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            has_bn = True
            break
    assert has_bn, "Model should use Batch Normalization layers"

def test_dropout_exists(model):
    has_dropout = False
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            has_dropout = True
            break
    assert has_dropout, "Model should use Dropout"

def test_gap_exists(model):
    has_gap = False
    for module in model.modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            has_gap = True
            break
    assert has_gap == False, "Model should not use Global Average Pooling (GAP)"

def test_accuracy():
    train_accuracy, test_accuracy = main()
    assert test_accuracy >= 95.0, f"Test accuracy is {test_accuracy}%, should be at least 95%"