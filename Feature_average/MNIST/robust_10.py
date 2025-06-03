import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


coarse_label_map = {}
for i in range(10):
    if i < 5:
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
    transforms.ToTensor(),  # 转为 Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化到 [-1, 1]
])


train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)



loss_function = nn.NLLLoss()


net = resnet18()
net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_classes = 10
net.fc = nn.Linear(net.fc.in_features, num_classes)
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/MNIST10_10.pth')
# new_state_dict = {}
# for k, v in checkpoint['net'].items():
#     if k.startswith('module.'):
#         new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
#     else:
#         new_state_dict[k] = v  # 保持原样
print("Keys in the checkpoint:")
print(checkpoint.keys())
net.load_state_dict(checkpoint['state_dict'])
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']




for i in range(10):
    learning_rate = 0.01
    tmp = 0.001
    alpha = tmp * (i+1) + tmp * 40
    epsilon = 4*alpha
    k = 5
    class LinfPGDAttack(object):
        def __init__(self, model,epsilon, alpha, k):
            self.model = model
            self.epsilon = epsilon
            self.alpha = alpha
            self.k = k  

        def perturb(self, x_natural, y):
            x = x_natural.detach()
            x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
            for i in range(k):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x)
                    probability = F.softmax(logits, dim=1)
                    probability = probability @ A
                    loss = loss_function(torch.log(probability),y)
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + alpha * torch.sign(grad.detach())
                x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
                
            return x
    class L2PGDAttack(object):
        def __init__(self, model, epsilon, alpha, iteration):
            self.model = model
            self.epsilon = epsilon
            self.alpha = alpha
            self.k = iteration
        def perturb(self, x_natural, y):
            x = x_natural.detach()
            delta = torch.randn_like(x)  # 生成与输入相同形状的标准正态分布随机数
            delta = delta / torch.norm(delta.view(delta.size(0), -1), dim=1).view(-1, 1, 1, 1)  # 标准化到L2单位球面
            delta = delta * epsilon  # 缩放到L2范数为epsilon的球面
            x = x_natural + delta
            for i in range(self.k):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x)
                    probability = F.softmax(logits, dim=1)
                    probability = probability @ A
                    loss = loss_function(torch.log(probability),y)
                    # loss = F.cross_entropy(logits, y)
                grad = torch.autograd.grad(loss, [x])[0]
                grad_norms = torch.norm(grad.view(x.shape[0], -1), p=2, dim=1) + 1e-8
                grad = grad / grad_norms.view(x.shape[0], 1, 1, 1)
                x = x.detach() + self.alpha * grad
                delta = x - x_natural
                delta_norms = torch.norm(delta.view(x.shape[0], -1), p=2, dim=1)
                factor = self.epsilon / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                delta  = delta * factor.view(-1, 1, 1, 1)
                # x = torch.clamp(delta + x_natural, min=-2.22, max=2.51).detach()
                x = (x_natural + delta).detach()
            return x


    # cudnn.benchmark = True

    adversary = LinfPGDAttack(net, epsilon, alpha, k)
    

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
                loss = criterion(outputs, targets)
                benign_loss += loss.item()

                adv = adversary.perturb(inputs, fine_to_coarse_tensor(targets).to(device))
                adv_outputs = net(adv)
                loss = criterion(adv_outputs, targets)
                # with open("experiment result/sub2_test_loss.txt", 'a') as file:
                #     file.write(str(loss.item())+"\n")
                adv_loss += loss.item()

                _, predicted = outputs.max(1)
                predicted = fine_to_coarse_tensor(predicted).to('cuda:0')
                targets = fine_to_coarse_tensor(targets).to('cuda:0')

                benign_correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0:
                    print('\nCurrent batch:', str(batch_idx))
                    print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                    # print('Current benign test loss:', loss.item())

                

                _, predicted = adv_outputs.max(1)
                predicted = fine_to_coarse_tensor(predicted).to('cuda:0')
                adv_correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0:
                    print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                    # print('Current adversarial test loss:', loss.item())

        print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
        print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
        print('Total benign test loss:', benign_loss)
        print('Total adversarial test loss:', adv_loss)

        # with open("attack_result/10_test_benign_accuracy.txt", 'a') as file:
        #     file.write(str(100. * benign_correct / total)+"\n")
        with open("attack_result_inf/MNIST_10_k=5.txt", 'a') as file:
            file.write(str(100. * adv_correct / total)+"\n")
        # with open("attack_result/10_test_benign_loss.txt", 'a') as file:
        #     file.write(str(benign_loss)+"\n")
        # with open("attack_result/10_test_adv_loss.txt", 'a') as file:
        #     file.write(str(adv_loss)+"\n")



    for epoch in range(0, 1):
        test(epoch)
        # train(epoch)
        
    
    