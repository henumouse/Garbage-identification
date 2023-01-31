# coding=utf-8
"""
训练模型
"""
import torch
from torch.optim import SGD
from torchvision import transforms
import cv2
from matplotlib import pyplot as plt
from trash2 import DataUtils
from torch.utils.data import DataLoader
from torch import nn
from trash3 import TrashNet


def a1():
    img = cv2.imread('../data/wkong.png')
    h, w, c = img.shape
    rs = cv2.resize(img, (h // 2, w // 2))
    rs = cv2.flip(img, 1)
    plt.imshow(rs)
    plt.show()


def fit():
    Image_size = 32
    norm_mean = [0.468, 0.456, 0.406]  # 平均值
    norm_std = [0.229, 0.224, 0.225]  # 表准差
    # 图片变换器(旋转伸缩等、数据增强）
    transformer = transforms.Compose([
        transforms.Resize((Image_size, Image_size)),
        transforms.RandomGrayscale(p=0.9),  # 图片数量少，数据量不够、可以通过变化尺寸形态
        # 这种图片变换一般是通过opencv
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    # 对训练集进行处理增强
    train = DataUtils(path="ga2\\train", transform=transformer)
    # 数据加载器
    trainloader = DataLoader(
        dataset=train,  # 数据集
        batch_size=20,  # 每次的尺寸
        shuffle=True,
        drop_last=True
    )
    # 建模
    model = TrashNet(classes=3)
    # 损失函数
    lossfn = nn.CrossEntropyLoss()
    # 优化函数 学习率lr=0.12
    opt = SGD(model.parameters(), lr=0.12)
    for e in range(40):
        for batch in trainloader:
            img, label = batch
            out = model(img)
            lossval = lossfn(out, label)
            opt.zero_grad()
            lossval.backward()
            opt.step()
    # 保存模型
    torch.save(model.state_dict(), "trash.pkl")


if __name__ == '__main__':
    fit()
