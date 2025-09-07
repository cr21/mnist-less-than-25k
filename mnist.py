from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0  # Initialize test_loss
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):  # Fix typo in batch_idx
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc=f'Loss={loss.item():0.4f} Batch_id={batch_idx}')
        pbar.set_postfix({'acc': f'{100*correct/processed:0.2f}%'})
    
    return 100. * correct / processed




def main():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params}')
    assert total_params <= 25_000, "Model should have less than 25000 parameters."
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    torch.manual_seed(1)
    batch_size=64
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Update the data loaders to include data augmentation
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,)),
                          transforms.RandomRotation(10),  # Move after normalization
                          transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Move after normalization
                      ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, 
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1

    for epoch in range(1, epochs + 1):
        train_accuracy = train(model, device, train_loader, optimizer, epoch)
        test_accuracy = test(model, device, test_loader)
        print(f'Epoch {epoch}: Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
    return train_accuracy, test_accuracy
if __name__ == "__main__":
    main()