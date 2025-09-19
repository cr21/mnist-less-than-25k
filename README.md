# MNIST
[![Python Test](https://github.com/cr21/mnist-less-than-25k/actions/workflows/test_model.yml/badge.svg)](https://github.com/cr21/mnist-less-than-25k/actions/workflows/test_model.yml)
## Objective
- Achieve >99.4 % test accuracy with below constraint
1. Total parameters should less than 25k
2. should Achieve > 99.4 %  test accuracy in 20 epoch


My Architecture:
```py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) #input -? OUtput? RF
        #nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1d_1 =  nn.Conv2d(32,8, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv1d_2 =  nn.Conv2d(32,8,1,stride=1)
        self.conv5 = nn.Conv2d(8, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 32, 3)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv1d_3 =  nn.Conv2d(32,8,1,stride=1)
        self.bn7 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(72,10)
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x):
        x = self.conv1d_1(self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))))
        x = self.dropout(x)
        x = self.conv1d_2(self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x))))))))
        x = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x))))))
        x = F.relu(self.bn7(self.conv1d_3(x)))
        x = x.view(x.size(0), -1) 
        return F.log_softmax(F.relu(self.fc(x)))

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
            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
            Conv2d-3           [-1, 32, 28, 28]           4,640
       BatchNorm2d-4           [-1, 32, 28, 28]              64
         MaxPool2d-5           [-1, 32, 14, 14]               0
            Conv2d-6            [-1, 8, 14, 14]             264
           Dropout-7            [-1, 8, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           1,168
       BatchNorm2d-9           [-1, 16, 14, 14]              32
           Conv2d-10           [-1, 32, 14, 14]           4,640
      BatchNorm2d-11           [-1, 32, 14, 14]              64
        MaxPool2d-12             [-1, 32, 7, 7]               0
           Conv2d-13              [-1, 8, 7, 7]             264
           Conv2d-14             [-1, 16, 5, 5]           1,168
      BatchNorm2d-15             [-1, 16, 5, 5]              32
           Conv2d-16             [-1, 32, 3, 3]           4,640
      BatchNorm2d-17             [-1, 32, 3, 3]              64
           Conv2d-18              [-1, 8, 3, 3]             264
      BatchNorm2d-19              [-1, 8, 3, 3]              16
           Linear-20                   [-1, 10]             730
================================================================
Total params: 18,242
Trainable params: 18,242
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.82
Params size (MB): 0.07
Estimated Total Size (MB): 0.89
----------------------------------------------------------------

```

- Created Github Action to test below test conditions

1. Total Parameter Count Test
2. Test Accuracy test


```log

❯ python3 mnist.py
Total Parameters: 18242
  0%|                                                                                   | 0/469 [00:00<?, ?it/s]/Users/chiragtagadiya/Downloads/MyProjects/ERA4/S4/mnist-less-than-25k/mnist.py:76: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(F.relu(self.fc(x)))
Loss=0.2707 Batch_id=468: 100%|███████████████████████████████████| 469/469 [01:53<00:00,  4.13it/s, acc=86.05%]

Test set: Average loss: 0.0719, Accuracy: 9790/10000 (97.9%)

Epoch 1: Train Accuracy: 86.05%, Test Accuracy: 97.90%
Loss=0.1460 Batch_id=468: 100%|███████████████████████████████████| 469/469 [01:57<00:00,  3.98it/s, acc=97.23%]

Test set: Average loss: 0.0468, Accuracy: 9870/10000 (98.7%)

Epoch 2: Train Accuracy: 97.23%, Test Accuracy: 98.70%
Loss=0.0976 Batch_id=468: 100%|███████████████████████████████████| 469/469 [01:59<00:00,  3.93it/s, acc=97.86%]

Test set: Average loss: 0.0320, Accuracy: 9911/10000 (99.1%)

Epoch 3: Train Accuracy: 97.86%, Test Accuracy: 99.11%
Loss=0.0361 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:00<00:00,  3.88it/s, acc=98.06%]

Test set: Average loss: 0.0327, Accuracy: 9906/10000 (99.1%)

Epoch 4: Train Accuracy: 98.06%, Test Accuracy: 99.06%
Loss=0.0710 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:05<00:00,  3.75it/s, acc=98.33%]

Test set: Average loss: 0.0284, Accuracy: 9910/10000 (99.1%)

Epoch 5: Train Accuracy: 98.33%, Test Accuracy: 99.10%
Loss=0.0348 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:06<00:00,  3.72it/s, acc=98.37%]

Test set: Average loss: 0.0232, Accuracy: 9928/10000 (99.3%)

Epoch 6: Train Accuracy: 98.37%, Test Accuracy: 99.28%
Loss=0.0637 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:09<00:00,  3.63it/s, acc=98.52%]

Test set: Average loss: 0.0442, Accuracy: 9869/10000 (98.7%)

Epoch 7: Train Accuracy: 98.52%, Test Accuracy: 98.69%
Loss=0.0282 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:11<00:00,  3.57it/s, acc=98.59%]

Test set: Average loss: 0.0262, Accuracy: 9917/10000 (99.2%)

Epoch 8: Train Accuracy: 98.59%, Test Accuracy: 99.17%
Loss=0.1299 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:11<00:00,  3.57it/s, acc=98.68%]

Test set: Average loss: 0.0231, Accuracy: 9924/10000 (99.2%)

Epoch 9: Train Accuracy: 98.68%, Test Accuracy: 99.24%
Loss=0.0187 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:10<00:00,  3.58it/s, acc=98.67%]

Test set: Average loss: 0.0270, Accuracy: 9916/10000 (99.2%)

Epoch 10: Train Accuracy: 98.67%, Test Accuracy: 99.16%
Loss=0.0498 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:08<00:00,  3.65it/s, acc=98.73%]

Test set: Average loss: 0.0201, Accuracy: 9942/10000 (99.4%)

Epoch 11: Train Accuracy: 98.73%, Test Accuracy: 99.42%
Loss=0.0375 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:00<00:00,  3.91it/s, acc=98.78%]

Test set: Average loss: 0.0191, Accuracy: 9936/10000 (99.4%)

Epoch 12: Train Accuracy: 98.78%, Test Accuracy: 99.36%
Loss=0.0294 Batch_id=468: 100%|███████████████████████████████████| 469/469 [01:59<00:00,  3.93it/s, acc=98.82%]

Test set: Average loss: 0.0227, Accuracy: 9932/10000 (99.3%)

Epoch 13: Train Accuracy: 98.82%, Test Accuracy: 99.32%
Loss=0.0094 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:00<00:00,  3.91it/s, acc=98.93%]

Test set: Average loss: 0.0221, Accuracy: 9939/10000 (99.4%)

Epoch 14: Train Accuracy: 98.93%, Test Accuracy: 99.39%
Loss=0.0244 Batch_id=468: 100%|███████████████████████████████████| 469/469 [01:59<00:00,  3.91it/s, acc=98.90%]

Test set: Average loss: 0.0237, Accuracy: 9936/10000 (99.4%)

Epoch 15: Train Accuracy: 98.90%, Test Accuracy: 99.36%
Loss=0.0355 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:03<00:00,  3.81it/s, acc=98.91%]

Test set: Average loss: 0.0212, Accuracy: 9939/10000 (99.4%)

Epoch 16: Train Accuracy: 98.91%, Test Accuracy: 99.39%
Loss=0.0326 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:01<00:00,  3.88it/s, acc=99.03%]

Test set: Average loss: 0.0169, Accuracy: 9946/10000 (99.5%)

Epoch 17: Train Accuracy: 99.03%, Test Accuracy: 99.46%
Loss=0.0196 Batch_id=468: 100%|███████████████████████████████████| 469/469 [02:03<00:00,  3.81it/s, acc=98.96%]

Test set: Average loss: 0.0269, Accuracy: 9913/10000 (99.1%)

Epoch 18: Train Accuracy: 98.96%, Test Accuracy: 99.13%
Loss=0.0054 Batch_id=468: 100%|███████████████████████████████████| 469/469 [01:46<00:00,  4.41it/s, acc=99.01%]

Test set: Average loss: 0.0210, Accuracy: 9939/10000 (99.4%)

Epoch 19: Train Accuracy: 99.01%, Test Accuracy: 99.39%
Loss=0.0396 Batch_id=468: 100%|███████████████████████████████████| 469/469 [01:49<00:00,  4.27it/s, acc=99.01%]

Test set: Average loss: 0.0204, Accuracy: 9945/10000 (99.5%)

Epoch 20: Train Accuracy: 99.01%, Test Accuracy: 99.45%

```