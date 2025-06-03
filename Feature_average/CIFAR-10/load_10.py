import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import CIFAR10
# from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
from cifar10_models.resnet import resnet18


# from torch.utils.data import Subset


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on:", device)



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


trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


net = resnet18(pretrained=True)
# num_ftrs = net.fc.in_features

# 修改最后两层
# net.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 512),  # 新的全连接层，将特征数从num_ftrs缩小到512
#     nn.ReLU(),                 # 添加ReLU激活函数
#     nn.Dropout(0.5),           # 添加Dropout层，防止过拟合
#     nn.Linear(512, 10)         # 最后的全连接层，将特征数从512缩小到目标类别数10
# )

# # 冻结前面的层
# for param in net.parameters():
#     param.requires_grad = False

# # 只训练最后的全连接层
# for param in net.fc.parameters():
#     param.requires_grad = True
# model_path = '/home/panzhixuan/CV/CIFAR-10/checkpoint/resnet18.pt'


# net.load_state_dict(torch.load(model_path))

net.to(device)
net.eval()
        

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = CosineAnnealingLR(optimizer, T_max=200)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)



# load checkpoint
# checkpoint_path = 'checkpoint/CIFAR10_2_200.pth'  
# checkpoint = torch.load(checkpoint_path)

# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

clusters = [[] for _ in range(10)]

training_loss = []
test_accuracy = []

from tqdm import tqdm
for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
    inputs, labels = data[0].to(device), data[1].to(device)
    features = None

    # 定义一个钩子函数来获取全连接层之前的输出
    def hook(module, input, output):
        global features
        features = output

    # 注册钩子到全连接层之前的平均池化层（avgpool层）
    hook_handle = net.avgpool.register_forward_hook(hook)

    # 前向传播
    with torch.no_grad():
        _ = net(inputs)

    # 注销钩子
    hook_handle.remove()
    features = torch.squeeze(features)
    for i in range (features.shape[0]):
        clusters[labels[i]].append(features[i])
        # optimizer.zero_grad()
        # # from IPython import embed; embed()
        # outputs = net(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # loss.backward(retain_graph=True)
        # optimizer.step()
        # running_loss += loss.item()

    # save checkpoint
    # if epoch%10 == 9:
    #     checkpoint = {
    #         'epoch': epoch + 1,
    #         'model_state_dict': net.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': running_loss / len(trainloader),
    #     }
    #     torch.save(checkpoint, f'checkpoint/pretrained_10_{epoch+1}.pth')
    
    # correct = 0
    # total = 0

    # for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
    #     inputs, labels = data[0].to(device), data[1].to(device)
    #     # print("input", inputs, "/n")
    #     optimizer.zero_grad()
    #     outputs = net(inputs)
    #     # from IPython import embed; embed()
    #     loss = criterion(outputs, labels)

    #     _, predicted = torch.max(outputs.data, 1)
    #     # predicted = fine_to_coarse_tensor(predicted).to('cuda:0')
        
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()


     
    # test_accuracy.append(correct / total)
    # print(f"Epoch:{epoch+1}, Training_loss:{running_loss/len(trainloader)}")
    # print(f"Accuracy on 10000 test images: {100 * correct / total}%")
means = []
for i in range(10):
    means.append(sum(clusters[i])/len(clusters[i]))
cpu_means = [tensor.to('cpu') for tensor in means]

num_ftrs = net.fc.in_features

net.fc = nn.Sequential(
    nn.Linear(num_ftrs, 30),  # 新的全连接层，将特征数从num_ftrs缩小到512
    nn.ReLU(),                 # 添加ReLU激活函数
    # nn.Dropout(0.5),           # 添加Dropout层，防止过拟合
    nn.Linear(30, 10)         # 最后的全连接层，将特征数从512缩小到目标类别数10
)

# load checkpoint
checkpoint_path = 'checkpoint/twolayer_10_10.pth'  
checkpoint = torch.load(checkpoint_path)

net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
weight = net.fc[0].weight.cpu()
# from IPython import embed; embed()
cos = [[] for _ in range(30)]
weights = [[] for _ in range(10)]
for i in range(30):
    for j in range(10):
        cosine=torch.dot(weight[i],cpu_means[j]).item()/(torch.norm(weight[i])*torch.norm(cpu_means[j])).item()
        cos[i].append(cosine)
    weights[int(i/3)].append(cos[i])

weights_mean = []
for i in range(10):
    weights_tensor = torch.tensor(weights[i])
    weights_mean.append(sum(weights_tensor)/len(weights_tensor))

matrix = np.array(weights_mean)

import matplotlib.pyplot as plt
from matplotlib import font_manager
font = font_manager.FontProperties(family='serif', weight='bold', size=18)
plt.imshow(matrix, cmap='viridis', interpolation='nearest')
cbar = plt.colorbar()

# 修改 colorbar 数字的字体大小
cbar.ax.tick_params(labelsize=18)
mu_labels = [r'$\boldsymbol{{\mu}}_{{{}}}$'.format(i) for i in range(1, 11)]
w_labels= [r'$\boldsymbol{{w}}_{{{}}}$'.format(i) for i in range(1, 11)]

plt.xticks(ticks=np.arange(10), labels=w_labels, fontproperties=font)
plt.yticks(ticks=np.arange(10), labels=mu_labels, fontproperties=font)
plt.show()
plt.savefig("figure/11.pdf", format='pdf')
# print(weights_mean)
        