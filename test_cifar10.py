import torch, torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from net import Model

batch_size = 500
use_cuda = torch.cuda.is_available()

net = Model()
if use_cuda:
    net.cuda()
state_dict = torch.load('model.pt')
net.load_state_dict(state_dict)

compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testset = torchvision.datasets.CIFAR10(root="./data", train=False,transform=compose)
print(len(testset))
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

test_accs = []

net.eval()
with torch.no_grad():
    for test_datas, test_labels in testloader:
        if use_cuda:
            test_datas, test_labels = test_datas.cuda(), test_labels.cuda()
        pred = net(test_datas)
        test_acc = pred.argmax(1).eq(test_labels).sum() / len(test_labels)
        test_accs.append(test_acc.item())
print(f"Test Acc: {np.mean(test_accs):.6f}")