from torch import nn
from torch.nn import functional as F

D = 1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Secret(nn.Module):
    def __init__(self):
        super(Secret, self).__init__()
        self.conv = nn.Sequential(       #(3,400,400)
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(3, 3), ),  # (64,  132,  132)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),  # (64,  44,  44)
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(3, 3), ),  # (32,  14,  14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32 * 4 * 4, 1000),
            nn.Linear(1000, D*400 * 400),
        )

    def forward(self, img):
        img = self.conv(img)
        img = img.view(img.shape[0], -1)
        out = self.fc(img)
        return out

#ResNet18
class Clf(nn.Module):
    def __init__(self, num_classes=10):
        super(Clf, self).__init__()
        self.block = BasicBlock
        self.num_blocks = [2, 2, 2, 2]
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 逐层搭建ResNet
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * self.block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # layers = [ ] 是一个列表
        # 通过下面的for循环遍历配置列表，可以得到一个由 卷积操作、池化操作等 组成的一个列表layers
        # return nn.Sequential(*layers)，即通过nn.Sequential函数将列表通过非关键字参数的形式传入(列表layers前有一个星号)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#组合消息提取器和分类器两个模型
# 通过forward方法根据传入的model_type参数来决定使用Secret还是Clf模型进行前向传播
#定义一个包含秘密消息提取和图像分类功能的神经网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.secret = Secret()
        self.clf = Clf()

    def forward(self, img, model_type="clf"):
        if model_type == "010001":
            return self.secret(img)
        elif model_type == "clf":
            return self.clf(img)
        else:
            raise ValueError("Invalid model type")


if __name__ == '__main__':
    model = Model()
    for name, param in model.named_parameters():
        print(name)