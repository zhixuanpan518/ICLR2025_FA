import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



coarse_label_map = {}

for i in range(10):
    if i % 2 == 0:
        coarse_label_map[i] = 0  
    else:
        coarse_label_map[i] = 1  



A = torch.zeros(10, 2)

# 根据coarse_label_map填充矩阵A
for i, j in coarse_label_map.items():
    A[i, j] = 1
A = A.to(device)

def fine_to_coarse(target):    
    return coarse_label_map[target]

def fine_to_coarse_tensor(fine_labels):
    coarse_labels = [coarse_label_map[label.item()] for label in fine_labels]
    return torch.tensor(coarse_labels)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform,target_transform=fine_to_coarse)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform, target_transform=fine_to_coarse)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)
net = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

    print(f"Epoch {epoch} - Training loss: {running_loss / len(train_loader):.6f}")


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"Test loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")


def save_checkpoint(state, filename="MNIST_10.pth"):
    torch.save(state, filename)


num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(model, train_loader, criterion, optimizer, epoch)
    test(model, test_loader, criterion)

    # 保存模型的checkpoint
    if epoch % 10 == 0:
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=f"./checkpoint/MNIST2_{epoch}.pth")

print("Training complete and checkpoints saved.")
