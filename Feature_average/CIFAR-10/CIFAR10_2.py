import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import Subset


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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



trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train, target_transform=fine_to_coarse)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test, target_transform=fine_to_coarse)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
#            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
#            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
#            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
#            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
#            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
#            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
#            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
#            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
#            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
#            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
#            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
#            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
#            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
#            'worm')

learning_rate = 0.1
epsilon = 0.00314 * 2
# epsilon = 0.000000000000000314
k = 7
alpha = 0.00784



# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])


# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)


# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# subset_size = 50000

# selected_indices = []

# with open('random_permutation.txt', "r") as file:
#     for i, line in enumerate(file):
#         index = int(line.strip())
#         selected_indices.append(index)

#         if i + 1 >= subset_size:
#             break
# subset = Subset(trainset, selected_indices)


# trainloader_subset = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True, num_workers=2)

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
                # coarse_probability = probability @ A
                # loss = loss_function(torch.log(coarse_probability),y)
                loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
        return x



# net = resnet18(pretrained=False)  
# net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# net.fc = torch.nn.Linear(net.fc.in_features, 10)
net = resnet18(pretrained=False, num_classes=2)
net.to(device)

        
adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = CosineAnnealingLR(optimizer, T_max=200)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
# # optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=5e-4)
# from torch.optim.lr_scheduler import CosineAnnealingLR

# optimizer = optim.Adam(net.parameters(), lr=5e-4)
# scheduler = CosineAnnealingLR(optimizer, T_max=50)  # 50 epochs为一个退火周期


# load checkpoint
# checkpoint_path = 'checkpoint/CIFAR10_2_200.pth'  
# checkpoint = torch.load(checkpoint_path)

# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']



training_loss = []
test_accuracy = []

from tqdm import tqdm
for epoch in range(200):
    running_loss = 0.0   
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

    # save checkpoint
    if epoch%20 == 19:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(trainloader),
        }
        torch.save(checkpoint, f'checkpoint/CIFAR10_2_{epoch+1}.pth')
    
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


        # split_logits = [outputs[i] for i in range(outputs.size(0))]
        # from IPython import embed; embed()
    #     for j in range(outputs.size(0)):
    #         sorted_logits = sorted(split_logits[j])
    #         for k in range (10):
    #             global_logit_epoch[k].append(sorted_logits[k].item())
    #         global_margin_epoch.append(sorted_logits[9].item()-sorted_logits[8].item())

    # global_avg_margin = sum(global_margin_epoch)/len(global_margin_epoch)
    # global_margin.append(global_avg_margin)
    # with open('experiment_data/global_margin.txt', 'a') as file:
    #     file.write(str(global_avg_margin)+"\n")

    # for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
    #     inputs, labels = data[0].to(device).requires_grad_(), data[1].to(device)
    #     outputs = net(inputs)
    #     loss = criterion(outputs, labels)
    #     # outputs = F.softmax(outputs, dim=1)
    #     split_logit = [outputs[i] for i in range(outputs.size(0))]
    #     for j in range(outputs.size(0)):
    #         sorted_logit = sorted(split_logit[j])
    #         for k in range(10):
    #             global_logit_epoch[k].append(sorted_logit[k].item())
    #     # loss = criterion(outputs, labels)
    #     outputs = F.softmax(outputs, dim=1)
    #     split_probability = [outputs[i] for i in range(outputs.size(0))]
    #     for j in range(outputs.size(0)):
    #         sorted_probability = sorted(split_probability[j])
    #         for k in range(10):
    #             global_probability_epoch[k].append(sorted_probability[k].item())
    #     loss = criterion(outputs, labels)
    
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data            
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = net(images)            
    #         _, predicted = torch.max(outputs.data, 1)
    #         # from IPython import embed; embed()
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
     
    test_accuracy.append(correct / total)
    print(f"Epoch:{epoch+1}, Training_loss:{running_loss/len(trainloader)}")
    print(f"Accuracy on 10000 test images: {100 * correct / total}%")
    # print(f"Local margin: {local_avg_margin}, Global margin: {global_avg_margin}%")
    # print(f"Local l derivative: {sum(local_avg_l_derivative)/len(local_avg_l_derivative)}, Global l derivative: {sum(global_avg_l_derivative)/len(global_avg_l_derivative)}")
    # for j in range(10):
    #     with open(f"experiment_data/local_probability_{j}.txt", 'a') as file:
    #         file.write(str(sum(local_probability_epoch[j])/len(local_probability_epoch[j]))+"\n")
    #     with open(f"experiment_data/global_probability_{j}.txt", 'a') as file:
    #         file.write(str(sum(global_probability_epoch[j])/len(global_probability_epoch[j]))+"\n")
    #     with open(f"experiment_data/local_logit_{j}.txt", 'a') as file:
    #         file.write(str(sum(local_logit_epoch[j])/len(local_logit_epoch[j]))+"\n")
    #     with open(f"experiment_data/global_logit_{j}.txt", 'a') as file:
    #         file.write(str(sum(global_logit_epoch[j])/len(global_logit_epoch[j]))+"\n")

    # with open('experiment_data/test_accuracy.txt', 'a') as file:
    #     file.write(str(correct / total)+"\n")


# 5. 测试模型
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy on 10000 test images: {100 * correct / total}%")



# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(local_margin, global_margin)
# plt.title('Local and Global Margin')
# plt.xlabel('Epoch')
# plt.ylabel('Margin')
# plt.savefig('figure/margin_gap',dpi=300)

