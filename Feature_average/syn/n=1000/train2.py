import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
# from models import *
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import Subset
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

train_data = np.load('train_data.npy')  
train_labels = np.load('train_label.npy')  
# train_labels = np.floor(np.arange(1000) / 100 ).astype(int)
test_data = np.load('test_data.npy')
test_labels = np.load('test_label.npy')
# test_labels = np.floor(np.arange(1000) / 100 ).astype(int)

train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size, bias=False)

        # self.layer1.weight = nn.Parameter(torch.tensor(weight1, dtype=torch.float32))
        self.layer2.weight = nn.Parameter(torch.tensor(weight1, dtype=torch.float32))

        nn.init.normal_(self.layer1.weight, mean=0.0, std= 0.00001)
        nn.init.normal_(self.layer1.bias, mean=0.0, std=0.0)

        # self.layer1.bias = nn.Parameter(torch.full((hidden_size,), -2.0))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

weight1 = [[1,1,1,1,1,-1,-1,-1,-1,-1]]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = TwoLayerNet(3072, 10, 1).to(device)

for param in net.layer2.parameters():
    param.requires_grad = False


class LogisticLoss(nn.Module):
    def __init__(self):
        super(LogisticLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.log(1 + torch.exp(-x * y))
        return loss.mean()  
    
criterion = LogisticLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

checkpoint_dir = 'checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

# def load_checkpoint(model, optimizer, checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")
#     return model, optimizer, epoch, loss

# checkpoint_path = 'checkpoint/checkpoint_epoch_5.pth' 
# model, optimizer, start_epoch, loss = load_checkpoint(net, optimizer, checkpoint_path)

for epoch in range(100):  
    net.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        # from IPython import embed; embed()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if (epoch+1) % 50 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'2test_{epoch+1}.pth')
        save_checkpoint(net, optimizer, epoch+1, running_loss/len(train_loader), checkpoint_path)
    print(f'Epoch [{epoch+1}/100], Loss: {running_loss/len(train_loader):.4f}')

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        outputs = outputs.squeeze()
        predicted = torch.sign(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')
