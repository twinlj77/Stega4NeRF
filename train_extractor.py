import torch, cv2, os, time
from PIL import Image
from datetime import datetime
from torchvision import transforms
from net import Model, D
# from model import Model, D
from torch import nn
import json
D=1
# net = resnet18()
# print(net)
# raise
dir = r"D:\Stega4NeRF\data\train_secret-view"
fils = os.listdir(dir)

lr = 1e-5
epochs = 2000

compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])


net = Model()

# 生成秘密
if os.path.exists("secret00.pt"):
    secret = torch.load("secret00.pt")
else:
    secret_shape = (1, D, 180, 180)
    secret = torch.zeros(secret_shape).random_(0, 2)
    secret = secret.view(secret.shape[0], -1)

# 加载模型
# if os.path.exists("modell2.pt"):
#     state_dict = torch.load('modell2.pt')
#     net.load_state_dict(state_dict)

# 冻结分类器参数，训练消息提取器
# for name, param in net.named_parameters():
#     if "secret" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

X_train = torch.zeros(size=(1, 3, 180, 180))
y_train = torch.zeros(size=(1, D * 180 * 180))
for idx, fil in enumerate(fils):
    filpath = os.path.join(dir, fil)
    # img = Image.open(filpath)
    # img = compose(img)
    img = cv2.imread(filpath)
    img = torch.from_numpy(img).to(torch.float32)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    print(img.shape)
    X_train[idx] = img
    y_train[idx] = secret[idx]

fil = open('output.log', 'a', encoding='utf8')
fil.write(
    "================train starting at {}==================\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


use_cuda = torch.cuda.is_available()

loss = nn.MSELoss()

optimizer = torch.optim.Adam([param for param in net.parameters() if param.requires_grad], lr=lr)

net.train()

for name, param in net.named_parameters():
    if param.requires_grad:
        print("可训练权重>>>", name)

if use_cuda:
    net.cuda()

count = 0
start_time = time.time()  # 开始时间
for epoch in range(epochs):

    if use_cuda:
        X_train, y_train = X_train.cuda(), y_train.cuda()
    net.zero_grad()
    out = net(X_train, model_type="010001")
    l = loss(out, y_train)
    decoder_acc = ((out >= 0.5).eq(y_train).sum(axis=1).float() / secret.shape[1]).mean()
    bpp = D * (2 * decoder_acc.item() - 1)
    if decoder_acc.item() == 1 and bpp == 1:
        end_time = time.time()  # 结束时间
        duration = int(end_time - start_time)
        print(f"Train acc=1 and Train bpp=2 achieved at epoch {epoch + 1}")
        print(f"Training time: {duration} s")
        # break
    # 输出结果
    json.dump({
        "epoch": epoch,
        "Loss": '%.6f' % l.item(),
        "train Acc": '%.6f' % decoder_acc.item(),
        "train bpp": '%.6f' % bpp,
    }, fil, ensure_ascii=False)
    fil.write('\n')
    print(f"Epoch {epoch} , Loss: {l.item()},  train Acc: {decoder_acc.item()}, train bpp:{bpp}  ")
    l.backward()
    optimizer.step()
    # if bpp == torch.tensor(1.0):
    #     count += 1
    #     print(11111, bpp)
    # if count > 10:
    #     break
fil.write("================train ending at {}==================\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
fil.close()
end_time = time.time()  # 结束时间

print(f"Training  duration:  {int(end_time - start_time)} s")  # 输出训练时长


state_dict = net.state_dict()

# 保存模型
torch.save(secret, "secret00.pt")
torch.save(state_dict, 'model00.pt')
