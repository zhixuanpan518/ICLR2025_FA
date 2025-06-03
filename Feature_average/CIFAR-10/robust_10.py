import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import os
from torchvision.models import resnet18

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# coarse_label_map = {
#         0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
#         10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
#         20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
#         30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
#         40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
#         50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
#         60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
#         70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
#         80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
#         90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13,
#     }
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




transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)



loss_function = nn.NLLLoss()


# from models import *


file_name = 'pgd_adversarial_training'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  
# ])



# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

net = resnet18(pretrained=False, num_classes=10)
net.to(device)





for i in range(20):
    learning_rate = 0.01 
    epsilon = 0.2* (i+1)
    # epsilon = 0.000000000000000314
    k = 7
    alpha = 0.05 * (i+1)
    class LinfPGDAttack(object):
        def __init__(self, model):
            self.model = model

        def perturb(self, x_natural, y):
            x = x_natural.detach()
            x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
            for i in range(k):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x)
                    # probability = F.softmax(logits, dim=1)
                    # # from IPython import embed; embed()
                    
                    # coarse_probability = torch.zeros(probability.shape[0],20).to(device)
                    # probability = probability @ A

                    # # for idx, val in enumerate(probability):
                    # #     # from IPython import embed; embed()
                    # #     for i in range(100):
                    # #         new_idx = coarse_label_map[i]
                    # #         coarse_probability[idx, new_idx] += val[i]
                    # # # from IPython import embed; embed()
                    # # print(coarse-coarse_probability)
                    # loss = loss_function(torch.log(probability),y)
                    # # from IPython import embed; embed()
                    loss = F.cross_entropy(logits, y)
                grad = torch.autograd.grad(loss, [x])[0]
                # from IPython import embed; embed()
                x = x.detach() + alpha * torch.sign(grad.detach())
                x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
                
            return x


    # class L2PGDAttack(object):
    #     def __init__(self, model):
    #         self.model = model

    #     def perturb(self, x_natural, y):
    #         x = x_natural.detach()
    #         # x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    #         for i in range(k):
    #             x.requires_grad_()
    #             with torch.enable_grad():
    #                 logits = self.model(x)
    #                 probability = F.softmax(logits, dim=1)
    #                 probability = probability @ A
    #                 loss = loss_function(torch.log(probability),y)
    #                 # loss = F.cross_entropy(logits, y)
    #             grad = torch.autograd.grad(loss, [x])[0]
    #             # x = x.detach() + alpha * torch.sign(grad.detach())
    #             x = x.detach() + alpha * grad.detach()
    #             # perturbation = x - x_natural
    #             delta = x - x_natural
    #             # from IPython import embed; embed()
    #             delta_norms = torch.norm(delta.view(x.shape[0], -1), p=2, dim=1)
    #             factor = epsilon / delta_norms
    #             factor = torch.min(factor, torch.ones_like(delta_norms))
                
    #             delta = delta * factor.view(-1, 1, 1, 1)
    #             # from IPython import embed; embed()
    #             x = (x_natural + delta).detach()
                
                



    #         return x
    class L2PGDAttack(object):
        def __init__(self, model, epsilon, alpha, iteration):
            self.model = model
            self.epsilon = epsilon
            self.alpha = alpha
            self.k = iteration

        def perturb(self, x_natural, y):
            x = x_natural.detach()
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
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


    # def attack(x, y, model, adversary):
    #     model_copied = copy.deepcopy(model)
    #     model_copied.eval()
    #     adversary.model = model_copied
    #     adv = adversary.perturb(x, y)
    #     return adv

    # net = ResNet18()
    # net = net.to(device)
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    adversary = L2PGDAttack(net, epsilon, alpha, k)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    checkpoint_path = 'checkpoint/CIFAR10_10_200.pth'
    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


    # def train(epoch):
    #     print('\n[ Train epoch: %d ]' % epoch)
    #     net.train()
    #     train_loss = 0
    #     correct = 0
    #     total = 0
    #     for batch_idx, (inputs, targets) in enumerate(train_loader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         optimizer.zero_grad()

    #         adv = adversary.perturb(inputs, targets)
    #         adv_outputs = net(adv)
    #         loss = criterion(adv_outputs, targets)
    #         loss.backward()

    #         optimizer.step()
    #         train_loss += loss.item()
    #         _, predicted = adv_outputs.max(1)

    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()
            
    #         if batch_idx % 10 == 0:
    #             print('\nCurrent batch:', str(batch_idx))
    #             print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
    #             print('Current adversarial train loss:', loss.item())

    #     print('\nTotal adversarial train accuarcy:', 100. * correct / total)
    #     print('Total adversarial train loss:', train_loss)

    def test(epoch):
        print('\n[ Test epoch: %d ]' % epoch)
        net.eval()
        benign_loss = 0
        adv_loss = 0
        benign_correct = 0
        adv_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
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

        with open("attack_result/10_test_benign_accuracy.txt", 'a') as file:
            file.write(str(100. * benign_correct / total)+"\n")
        with open("attack_result/10_test_adv_accuracy.txt", 'a') as file:
            file.write(str(100. * adv_correct / total)+"\n")
        with open("attack_result/10_test_benign_loss.txt", 'a') as file:
            file.write(str(benign_loss)+"\n")
        with open("attack_result/10_test_adv_loss.txt", 'a') as file:
            file.write(str(adv_loss)+"\n")


        # state = {
        #     'net': net.state_dict()
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/' + file_name)
        # print('Model Saved!')

    def train(epoch):
        print('\n[ train epoch: %d ]' % epoch)
        net.eval()
        benign_loss = 0
        adv_loss = 0
        benign_correct = 0
        adv_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # from IPython import embed; embed()
                total += targets.size(0)

                outputs = net(inputs)
                loss = criterion(outputs, targets)
                benign_loss += loss.item()

                adv = adversary.perturb(inputs, fine_to_coarse_tensor(targets).to(device))
                adv_outputs = net(adv)
                loss = criterion(adv_outputs, targets)
                # with open("experiment result/sub2_train_loss.txt", 'a') as file:
                #     file.write(str(loss.item())+"\n")
                adv_loss += loss.item()

                _, predicted = outputs.max(1)
                predicted = fine_to_coarse_tensor(predicted).to('cuda:0')
                targets = fine_to_coarse_tensor(targets).to('cuda:0')
                benign_correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0:
                    print('\nCurrent batch:', str(batch_idx))
                    print('Current benign train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                    # print('Current benign train loss:', loss.item())

                

                _, predicted = adv_outputs.max(1)
                predicted = fine_to_coarse_tensor(predicted).to('cuda:0')
                adv_correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0:
                    print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                #     print('Current adversarial train loss:', loss.item())

        print('\nTotal benign train accuarcy:', 100. * benign_correct / total)
        print('Total adversarial train Accuarcy:', 100. * adv_correct / total)
        print('Total benign train loss:', benign_loss)
        print('Total adversarial train loss:', adv_loss)

        with open("attack_result/10_train_benign_accuracy.txt", 'a') as file:
            file.write(str(100. * benign_correct / total)+"\n")
        with open("attack_result/10_train_adv_accuracy.txt", 'a') as file:
            file.write(str(100. * adv_correct / total)+"\n")
        with open("attack_result/10_train_benign_loss.txt", 'a') as file:
            file.write(str(benign_loss)+"\n")
        with open("attack_result/10_train_adv_loss.txt", 'a') as file:
            file.write(str(adv_loss)+"\n")


    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(0, 1):
        adjust_learning_rate(optimizer, epoch)
        test(epoch)
        train(epoch)
        
    
    