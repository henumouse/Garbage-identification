# coding=utf-8
"""建模-卷积神经网络"""
import torch
import torch.nn as nn
from torch.nn import functional as F


def a1():
    x = torch.randn([3, 1, 5, 4])  # 创建一个张量
    # print(type(x))
    # print(x.shape)
    # print(x)
    '''
    <class 'torch.Tensor'> 张量tensor
    torch.Size([3, 1, 5, 4]) 3通道 4维矩阵'''
    # 卷积输入数据一个特征，输出数据4个特征  用2行3列的矩阵与原始矩阵进行计算
    # 卷积核（2,3）宽是2 高是3 每次取2个数据
    conv = torch.nn.Conv2d(1, 2, (2, 3))
    # 卷积计算
    rs = conv(x)
    print(x)


# 卷积神经网络模型  继承已经存在的类
class TrashNet(torch.nn.Module):
    def __init__(self, classes):
        # 调用父类构造函数
        super(TrashNet, self).__init__()
        # 创建第一个卷积层：特征输入数据哟3个特征，输出特征有6个
        # 卷积核宽高都是5 （5,5）
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        # 第二层卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层 一般都是线性函数 全连接第一层 整个神经网络的第三层
        # 输入16*5*5,输出120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 第4层
        self.fc2 = nn.Linear(120, 84)
        # 第5层 最后一层 输出层
        self.fc3 = nn.Linear(84, classes)

    # 把这些层组合在一起进行计算
    def forward(self, x):
        # 对第一层卷积计算的结果使用relu激活函数
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # 输出图片的类别
        out = self.fc3(out)
        return out


if __name__ == '__main__':
    model = TrashNet(classes=2)
    img = torch.randn(1, 3, 32, 32)
    rs = model(img)
    # print(rs) 张量tensor([[ 0.0496, -0.0927]],classes是2所以tensor数组里是2个值 梯度函数 优化函数grad_fn=<AddmmBackward0>
    _, pred = torch.max(rs, dim=1)
    rs = pred.data.item()
    print(rs)  # 1 输出类别1
