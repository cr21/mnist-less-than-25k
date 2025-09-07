# MNIST
[![Python Test](https://github.com/cr21/mnist-less-than-25k/actions/workflows/test_model.yml/badge.svg)](https://github.com/cr21/mnist-less-than-25k/actions/workflows/test_model.yml)
## Objective
- Achieve >95 % test accuracy with below constraint
1. Total parameters should less than 25k
2. should Achieve > 95 %  test accuracy in 1 epoch


My Architecture:
```py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)       # (1,28,28) -> (8,26,26)
        self.bn1   = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, 3)      # (8,26,26) -> (16,24,24)
        self.bn2   = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3)     # (16,24,24) -> (32,22,22)
        self.bn3   = nn.BatchNorm2d(32)

        # Adaptive pooling to ensure output is always (32,4,4)
        self.gap   = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1   = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)        
        
        # (32,22,22) -> (32,4,4)
        x = x.view(x.size(0), -1)  
        # Flatten to (batch, 512)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

```

```py

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
       BatchNorm2d-2            [-1, 8, 26, 26]              16
            Conv2d-3           [-1, 16, 24, 24]           1,168
       BatchNorm2d-4           [-1, 16, 24, 24]              32
            Conv2d-5           [-1, 32, 22, 22]           4,640
       BatchNorm2d-6           [-1, 32, 22, 22]              64
 AdaptiveAvgPool2d-7             [-1, 32, 4, 4]               0
            Linear-8                   [-1, 10]           5,130
================================================================
Total params: 11,130
Trainable params: 11,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.46
Params size (MB): 0.04
Estimated Total Size (MB): 0.51
----------------------------------------------------------------
```

- Created Github Action to test below test conditions

1. Total Parameter Count Test
2. Test Accuracy test

```log
❯ python3 mnist.py
Total Parameters: 11130
Loss=0.0567 Batch_id=937: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [01:07<00:00, 13.88it/s, acc=89.66%]

Test set: Average loss: 0.0624, Accuracy: 9818/10000 (98.2%)

Epoch 1: Train Accuracy: 89.66%, Test Accuracy: 98.18%
```
