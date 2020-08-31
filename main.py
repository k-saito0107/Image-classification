import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#データセットの作成
from data_transform import Resize
from torchvision.datasets import ImageFolder
width = 512
height = 384
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)


transform = transforms.Compose([
    Resize(width, height),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


train_images=ImageFolder(
    'kw_resources/画像分類/data/Images/',
    transform=transform
)

test_images=ImageFolder(
    'kw_resources/画像分類/data/Annotations/',
    transform=transform
)

train_loader=torch.utils.data.DataLoader(train_images,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_images,batch_size=64,shuffle=True)


from model import ResNet
in_ch = 3
f_out = 64
n_ch = 67

model = ResNet(in_ch, f_out, n_ch)



#モデルの学習
from train import train
num_epoch = 1000

up_model = train(model, num_epoch, train_loader, test_loader)

torch.save(model.state_dict(),'kw_resources/画像分類/weights/resnet'+str(num_epoch)+'.pth')

print('finish')









