import torch, cv2, numpy as np
from PIL import Image
from torchvision import transforms
from net import Model, D
# from model import Model, D
from torch import nn

path = r'C:\Users\123\Desktop\Stega4NeRF\data\test_view_rotate\rotate_theta+5,phi-5.jpg'
# path = r'E:\180extractor_classify\data\train\origin.jpg'

# compose = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# img = Image.open(path)
# img = compose(img)

img = cv2.imread(path)
img = torch.from_numpy(img).to(torch.float32)
img = img.permute(2, 0, 1)

img = img.unsqueeze(0)
net = Model()
net.eval()
secret = torch.load('secret.pt')
print(secret)
state_dict = torch.load('model.pt')
net.load_state_dict(state_dict)
out = net(img, model_type="010001")
out  =  (out  >=  0.5).int()
print(out)
decoder_acc = ((out >= 0.5).eq(secret).sum(axis=1).float() / secret.shape[1]).mean()
print("准确率 ---> ", decoder_acc.item())
bpp = D * (2*decoder_acc.item()-1)
print("bpp ---> ", bpp)
