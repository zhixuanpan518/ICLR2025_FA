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
# train_labels = np.load('train_label.npy')  
train_labels = np.floor(np.arange(1000) / 100 ).astype(int)
test_data = np.load('test_data.npy')
# test_labels = np.load('test_label.npy')
test_labels = np.floor(np.arange(1000) / 100 ).astype(int)

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
        self.layer2.weight = nn.Parameter(torch.tensor(matrix.T, dtype=torch.float32))

        nn.init.normal_(self.layer1.weight, mean=0.0, std= 0.00001)
        nn.init.normal_(self.layer1.bias, mean=0.0, std=0.0)

        # self.layer1.bias = nn.Parameter(torch.full((hidden_size,), -2.0))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

num = 1
matrix = torch.zeros((10*num, 10))
for i in range(10):
    matrix[num*i:num*i+num, i] = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = TwoLayerNet(3072, 10*num, 10).to(device)

for param in net.layer2.parameters():
    param.requires_grad = False


class LogisticLoss(nn.Module):
    def __init__(self):
        super(LogisticLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.log(1 + torch.exp(-x * y))
        return loss.mean()
    
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss

checkpoint_path = 'checkpoint/10_100.pth' 
model, optimizer, start_epoch, loss = load_checkpoint(net, optimizer, checkpoint_path)

for i in range(100):
    epsilon = 2 * (i+1)
    k = 7
    alpha = 0.5 * (i+1)
    # class LinfPGDAttack(object):
    #     def __init__(self, model,epsilon, alpha, k):
    #         self.model = model
    #         self.epsilon = epsilon
    #         self.alpha = alpha
    #         self.k = k  

    #     def perturb(self, x_natural, y):
    #         x = x_natural.detach()
    #         x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    #         for i in range(k):
    #             x.requires_grad_()
    #             with torch.enable_grad():
    #                 logits = self.model(x)
    #                 probability = F.softmax(logits, dim=1)
    #                 probability = probability @ A
    #                 loss = loss_function(torch.log(probability),y)
    #             grad = torch.autograd.grad(loss, [x])[0]
    #             x = x.detach() + alpha * torch.sign(grad.detach())
    #             x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)                
    #         return x
        

    class L2PGDAttack(object):
        def __init__(self, model, epsilon, alpha, iteration):
            self.model = model
            self.epsilon = epsilon
            self.alpha = alpha
            self.k = iteration
        def perturb(self, x_natural, y):
            x = x_natural.detach()
            delta = torch.randn_like(x)  # 生成与输入相同形状的标准正态分布随机数
            delta = delta / torch.norm(delta.view(delta.size(0), -1), dim=1).view(-1, 1)  # 标准化到L2单位球面
            delta = delta * epsilon  # 缩放到L2范数为epsilon的球面
            x = x_natural + delta
            for i in range(self.k):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x)
                    # logits = logits.squeeze()
                    # from IPython import embed; embed()
                    # loss = criterion(logits, y)
                    loss = criterion(logits, y)
                    
                grad = torch.autograd.grad(loss, [x])[0]
                grad_norms = torch.norm(grad.view(x.shape[0], -1), p=2, dim=1) + 1e-8
                grad = grad / grad_norms.view(x.shape[0], 1)
                x = x.detach() + self.alpha * grad
                delta = x - x_natural
                delta_norms = torch.norm(delta.view(x.shape[0], -1), p=2, dim=1)
                factor = self.epsilon / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                delta  = delta * factor.view(-1, 1)
                # x = torch.clamp(delta + x_natural, min=-2.22, max=2.51).detach()
                x = (x_natural + delta).detach()
            return x

    adversary = L2PGDAttack(net, epsilon, alpha, k)

    def test(epoch):
        print('\n[ Test epoch: %d ]' % epoch)
        net.eval()
        benign_loss = 0
        adv_loss = 0
        benign_correct = 0
        adv_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)            
                total += targets.size(0)

                outputs = net(inputs)
                outputs = outputs.squeeze()
            
                loss = criterion(outputs, targets)
                # with open("experiment result/super2_benign_loss.txt", 'a') as file:
                #     file.write(str(loss.item())+"\n")
                benign_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                # predicted = fine_to_coarse_tensor(predicted).to('cuda:0')
                # targets = fine_to_coarse_tensor(targets).to('cuda:0')

                benign_correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0:
                    print('\nCurrent batch:', str(batch_idx))
                    print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                    print('Current benign test loss:', loss.item())
                targets_01 = (targets == 1).float()
                adv = adversary.perturb(inputs, targets)
                adv_outputs = net(adv)
                adv_outputs = adv_outputs.squeeze()
                # from IPython import embed; embed()
                loss = criterion(adv_outputs, targets)
                # with open("experiment result/super2_test_loss.txt", 'a') as file:
                #     file.write(str(loss.item())+"\n")
                adv_loss += loss.item()

                _, predicted = torch.max(adv_outputs, 1)
                # predicted = fine_to_coarse_tensor(predicted).to('cuda:0')
                adv_correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0:
                    print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                    print('Current adversarial test loss:', loss.item())

        print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
        print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
        print('Total benign test loss:', benign_loss)
        print('Total adversarial test loss:', adv_loss)

        with open("attack_result/10.txt", 'a') as file:
            file.write(str(100. * adv_correct / total)+"\n")


    for epoch in range(0, 1):
        test(epoch)