import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import os
from PIL import Image


def main():
    #データセットの作成
    from torchvision.datasets import ImageFolder
    from data_augumentation import Compose, Resize, Scale, RandomRotation, RandomMirror

    width = 512
    height = 512
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)


    train_transform = transforms.Compose([
        RandomRotation(angle=[-10,10]),
        RandomMirror(),
        Resize(width, height),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        Resize(width, height),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    train_images=ImageFolder(
        '/kw_resources/Img_classification/data/train/',
        transform=train_transform
    )

    test_images=ImageFolder(
        '/kw_resources/Img_classification/data/val/',
        transform=val_transform
    )

    train_loader=torch.utils.data.DataLoader(train_images,batch_size=12,shuffle=True, num_workers=2)
    test_loader=torch.utils.data.DataLoader(test_images,batch_size=12,shuffle=True, num_workers=2)


    from model import ResNet
    in_ch = 3
    f_out = 32
    n_ch = 37

    model = ResNet(in_ch, f_out, n_ch)



    #モデルの学習
    from train import train
    num_epoch = 300

    up_model = train(model, num_epoch, train_loader, test_loader)



print('finish')



if __name__ == '__main__':
    main()





