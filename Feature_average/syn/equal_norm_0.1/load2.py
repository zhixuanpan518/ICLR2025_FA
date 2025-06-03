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

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss

checkpoint_path = 'checkpoint/2test_100.pth' 
model, optimizer, start_epoch, loss = load_checkpoint(net, optimizer, checkpoint_path)



cluster = np.load('cluster.npy')
cluster=torch.tensor(cluster,dtype=torch.float32)

weight = net.layer1.weight.cpu()

cos = [[] for _ in range(10)]
weights = [[] for _ in range(10)]
for i in range(10):
    for j in range(10):
        cosine=torch.dot(weight[i],cluster[:,j]).item()/(torch.norm(weight[i])*torch.norm(cluster[:,j])).item()
        cos[i].append(cosine)
#     weights[int(i/10)].append(cos[i])

# weights_mean = []
# for i in range(10):
#     weights_tensor = torch.tensor(weights[i])
#     weights_mean.append(sum(weights_tensor)/len(weights_tensor))

matrix = np.array(cos)

import matplotlib.pyplot as plt
from matplotlib import font_manager
font = font_manager.FontProperties(family='serif', weight='bold', size=18)
plt.imshow(matrix, cmap='viridis', interpolation='nearest',aspect='equal')
plt.title('min norm/max norm=0.8/1.2', fontsize=14)
cbar = plt.colorbar()

cbar.ax.tick_params(labelsize=18)
mu_labels = [r'$\boldsymbol{{\mu}}_{{{}}}$'.format(i) for i in range(1, 11)]
w_labels= [r'$\boldsymbol{{w}}_{{{}}}$'.format(i) for i in range(1, 11)]

plt.xticks(ticks=np.arange(10), labels=w_labels, fontproperties=font)
plt.yticks(ticks=np.arange(10), labels=mu_labels, fontproperties=font)
plt.show()
plt.savefig("figure/Average.pdf", format='pdf')