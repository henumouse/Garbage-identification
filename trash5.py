# coding=utf-8
"""识别
"""
import torch
from PIL import Image
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
from trash3 import TrashNet


def reg():
    raw = Image.open("ga2/test/trash/trash48.jpg").convert('RGB')  # 图片彩色转成黑白
    new = F.resize(raw, size=(32, 32))  # 改变图片大小交给模型的都是长宽为32
    new = F.to_tensor(new)  # 转化成张量
    norm_mean = [0.485, 0.456, 0.406]  # 均值
    norm_std = [0.229, 0.224, 0.225]  # 标准差
    new = F.normalize(new, norm_std, norm_std)  # norm_mean,norm_std作为两个参数传过来对他进行正则化处理
    # 上面三个F函数都是对图片进行各种各样的变换
    # 把数据扩张为四维的矩阵
    new = new.expand(1, 3, 32, 32)  # 3通道
    # --加载模型--
    model = TrashNet(classes=3)
    model.load_state_dict(torch.load("trash.pkl"))
    # 识别垃圾类型
    out = model(new)  # 把图片new交给模型model进行预测识别
    _, pred = torch.max(out, dim=1)
    rs = pred.data.item()
    msg = ""
    if rs == 0:
        msg = "纸张垃圾"
    if rs == 1:
        msg = "塑料垃圾"
    if rs == 2:
        msg = "生活垃圾"

    print(msg)
    plt.imshow(raw)


if __name__ == '__main__':
    reg()
