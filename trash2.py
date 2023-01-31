# coding=utf-8
"""给图片打标签"""
import os
from PIL import Image
from torch.utils.data import Dataset


class DataUtils(Dataset):
    def __init__(self, path="", transform=None):
        # 存储图片名称和所属类别
        self.img_info = []
        for root, child, files in os.walk(path):
            for file in files:
                # 所有图片的路径
                fname = os.path.join(root, file)
                # print(fname) ga2\train\trash\trash71.jpg

                # 图片的标签
                label = fname.split("\\")[2]
                label2 = 0
                # print(label)
                if label == "paper":
                    label2 = 0
                if label == "plastic":
                    label2 = 1
                if label == "trash":
                    label2 = 2
                self.img_info.append([fname, label2])
        self.transform = transform
        # print(self.img_info)#[['ga2\\train\\paper\\paper1.jpg', 0],

    # 返回变换后的每一张图片
    def __getitem__(self, item):
        # 取出每一张图片和类别
        fname, label = self.img_info[item]
        img = Image.open(fname)
        # 图片转换 由彩色转为黑白
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # 图片数量
    def __len__(self):
        return len(self.img_info)


if __name__ == '__main__':
    x = DataUtils(path="ga2\\train")
    # \是转义字符 前面要用反斜杠
