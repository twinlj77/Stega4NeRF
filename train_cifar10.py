import torch
import torchvision
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from net import Model
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

batch_size = 128
lr = 0.1
epochs = 500 # Set the number of epochs as needed

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

net = Model()

if os.path.exists("model.pt"):
    state_dict = torch.load('model.pt')
    net.load_state_dict(state_dict)

# for name, param in net.named_parameters():
#     if "clf" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

use_cuda = torch.cuda.is_available()

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
net.train()
if use_cuda:
    net.cuda()

# Initialize lists to store losses and accuracies
train_losses = []
test_losses = []
train_accs = []
test_accs = []
best_test_acc = 0.0  # Track the best test accuracy
best_model_state = None
for epoch in range(epochs):
    net.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for step, (datas, labels) in enumerate(trainloader):
        if use_cuda:
            datas, labels = datas.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(datas)
        batch_loss = loss(outputs, labels)  # Fix: use a different variable name
        batch_loss.backward()
        optimizer.step()

        running_train_loss += batch_loss.item()

        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

        # if step % print_step == 0:
        #     print(f"Epoch: {epoch}, Step: {step}, Train Loss: {running_train_loss / (step + 1):.6f}")

    train_acc = correct_train / total_train
    train_losses.append(running_train_loss / len(trainloader))
    train_accs.append(train_acc)

    # Validation (test) phase
    net.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for test_datas, test_labels in testloader:
            if use_cuda:
                test_datas, test_labels = test_datas.cuda(), test_labels.cuda()
            test_outputs = net(test_datas)
            test_loss = loss(test_outputs, test_labels)
            running_test_loss += test_loss.item()

            _, test_predicted = test_outputs.max(1)
            total_test += test_labels.size(0)
            correct_test += test_predicted.eq(test_labels).sum().item()

    test_acc = correct_test / total_test
    test_losses.append(running_test_loss / len(testloader))
    test_accs.append(test_acc)

    print(f"Epoch: {epoch}, Train Loss: {train_losses[-1]:.6f}, Train Acc: {train_acc:.6f}, Test Loss: {test_losses[-1]:.6f}, Test Acc: {test_acc:.6f}")

    # Check if the current model has the best test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_state = net.state_dict()
        torch.save(best_model_state,'model.pt')
    scheduler.step()

# if best_model_state is not None:
#     torch.save(best_model_state, 'model3.pt')
# Plot and save training and testing metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('metrics.png')
plt.show()

# Plot and save confusion matrix
net.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for test_datas, test_labels in testloader:
        if use_cuda:
            test_datas, test_labels = test_datas.cuda(), test_labels.cuda()
        pred = net(test_datas)
        all_predictions.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())
net.train()

conf_matrix = confusion_matrix(all_labels, all_predictions)
class_names = testset.classes

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')
plt.show()
