import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from cifar10_models.resnet import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import Subset


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

loss_function = nn.NLLLoss()

# coarse_label_map = {
#     0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
#     10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
#     20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
#     30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
#     40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
#     50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
#     60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
#     70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
#     80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
#     90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13,
# }
coarse_label_map = {}
for i in range(10):
    if i < 5 :
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



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])


trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train, target_transform=fine_to_coarse)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test, target_transform=fine_to_coarse)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)



net = ResNet2_18()
num_ftrs = net.linear.in_features

# 修改最后两层
net.linear = nn.Sequential(
    nn.Linear(num_ftrs, 100),  # 新的全连接层，将特征数从num_ftrs缩小到512
    nn.ReLU(),                 # 添加ReLU激活函数
    # nn.Dropout(0.5),           # 添加Dropout层，防止过拟合
    nn.Linear(100, 2)         # 最后的全连接层，将特征数从512缩小到目标类别数10
)

matrix = torch.zeros((100, 2))
for i in range(2):
    matrix[50*i:50*i+50, i] = 1

with torch.no_grad():
    net.linear[2].weight = nn.Parameter(matrix.t())
    net.linear[2].bias = nn.Parameter(torch.zeros(2))


for param in net.parameters():
    param.requires_grad = False

for param in net.linear.parameters():
    param.requires_grad = True

for param in net.linear[2].parameters():
    param.requires_grad = False

net.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)




training_loss = []
test_accuracy = []

from tqdm import tqdm
for epoch in range(20):
    running_loss = 0.0   
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    # save checkpoint
    if epoch%10 == 9:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(trainloader),
        }
        torch.save(checkpoint, f'checkpoint/twolayer_2_{epoch+1}.pth')
    
    correct = 0
    total = 0

    for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
        inputs, labels = data[0].to(device), data[1].to(device)
        # print("input", inputs, "/n")
        optimizer.zero_grad()
        outputs = net(inputs)
        # from IPython import embed; embed()
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        # predicted = fine_to_coarse_tensor(predicted).to('cuda:0')
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


     
    test_accuracy.append(correct / total)
    print(f"Epoch:{epoch+1}, Training_loss:{running_loss/len(trainloader)}")
    print(f"Accuracy on 10000 test images: {100 * correct / total}%")

