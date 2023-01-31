# coding=utf-8
# 自然语言处理 之前是，现在是计算机视觉
"""分割数据集"""
import os
# 目录和文件操作
import random
import shutil


def a1():
    # 遍历这个目录下的所有子目录文件
    for root, child, files in os.walk("ga"):
        # print(root) 当前目录
        # print(child)子目录
        # print(files)所有文件
        for file in child:
            fname = os.path.join(root, file)
            print(fname)


# 分割数据集
def filesplit():
    dataroot = "ga2\\"
    for root, child, files in os.walk("ga"):
        for ch in child:
            childpath = os.path.join(root, ch)  # ga\paper ga\plastic ga\trash
            # 遍历目录下所有文件
            fname = os.listdir(childpath)
            # 太规整了不好，打乱顺序
            random.shuffle(fname)
            # 文件数量
            cfiles = len(fname)
            train = cfiles * 0.9
            test = cfiles * 0.1
            for index, f in enumerate(fname):
                img_name = os.path.join(childpath, f)

                # print(img_name)
                # ga\paper\paper353.jpg
                # ga\plastic\plastic6.jpg
                # ga\trash\trash130.jpg
                label = childpath.split("\\")[-1]
                # print(label)  trash paper plastic
                if index < train:
                    datapath = dataroot + "train\\" + label
                else:
                    datapath = dataroot + "\\test\\" + label
                    # 创建目录
                os.makedirs(datapath, exist_ok=True)
                datapath = datapath + "\\" + f
                shutil.copy(img_name, datapath)


if __name__ == '__main__':
    filesplit()
