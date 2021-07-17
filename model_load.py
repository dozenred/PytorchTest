import torch
import os  # 一定要有这个设定，要不然报错，也生成不了图

from lesson10_4 import Net

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 以上两行代码是为了解决OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.这个问题的
import torch
# dataset dataloader是之前讲过的两个用于导入数据的包 transforms是torchvision中对图像处理的工具
# torchvision.transform的功能是对PIL.Image进行变换
# torchvision.datasets 包含了若干数据集，包括MNIST
# torch.utils.data.DataLoader 的作用是PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
# torch.nn.funtional 包含了大量构造模型所需要的函数
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# 全链接层的激活不再使用sigmoid，改用ReLU
import torch.nn.functional as F
# 使用torch提供的优化器
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import argparse

# 这里用来放置画图用到的列表
epoch_list = []
accuracy_list = []  # 精度在测试集中
loss_list = []  # loss在训练集中
f1_list = []  # 暂时不会

batch_size = 64
# pytorch中读取图像时使用的是PythonImageLibrary，图像的数据是28*28像素再乘每个像素点的通道值，即0到255
# 一般读取的是Weight*Height*Channel，在pytorch中要将其转换为Channel*Weight*Height
# Compose可以把中括号里一系列transform组合起来使用，先用totensor转换为pytorch的张量，即c*w*h，
# ToTensor的源码：Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
#     Converts a PIL Image or numpy.ndarray (H x W x C) in the range
#     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
# 翻译一下，就是将shape为(h*w*c)的img转换为shape为(c*h*w)的tensor，并将其每个数值归一化到[0,1]
# 下一步Normalize是标准化数据，刚刚的ToTensor把灰度从[0,255]转换为[0,1.0]，然后利用Normalize将灰度的数据将呈现为均值为0方差为1的数据分布
# 1307和3081是整个数据集的均值和标准差

transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./dataset/',  # 选择数据集保存路径
                               train=True,  # 该数据集是否是训练集
                               download=False,  # 是否需要下载
                               transform=transfrom)  # 确定转换为张量并归一化的流程

test_dataset = datasets.MNIST(root='./dataset/',
                              train=False,
                              download=False,
                              transform=transfrom)
# DataLoader 数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器
train_loader = DataLoader(train_dataset,
                          shuffle=True,  # 是否在每个epoch重新打乱数据
                          batch_size=batch_size)  # 每个batch里有多少个样本

test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# 输入的维度是个4阶张量
# 即N个样本，通道值，w，h
# 全连接神经网络要求输入的样本是个矩阵
# 所以要将1*28*28的三阶张量变成1阶向量

# 这里用到view函数 即x=x.view(batch_size2, -1)
# 其中batch_size2就是存放照片的数目
# 这里面有两个参数，也就是说把它变成二阶张量，即矩阵，照片数目那么多行。第2个参数是-1，也就是说将来会自动计算有多少列，这个行数就是样本数N
# 后面Debug可以看到在使用view函数后，x的shape为64*320
# 这个x（现在是64*320的矩阵形式），将经过线性层，映射到10个结果上
# 最后经过线性层变成10个，也就是从0到9预测的”概率“
def get_parser():
    parser = argparse.ArgumentParser(description='为模型配置参数')
    parser.add_argument('--epoch', type=int, help='指定训练与测试模型的epoch，默认为10', default=10)
    parser.add_argument('--conv1st_in', type=int, help='指定首个卷积层的输入通道大小，默认为1', default=1)
    parser.add_argument('--conv1st_out', type=int, help='指定首个卷积层的输出通道大小，默认为10', default=10)
    parser.add_argument('--conv1st_size', type=int, help='指定首个卷积层的卷积核大小，默认为5', default=5)
    parser.add_argument('--conv2nd_in', type=int, help='指定第二个卷积层的输入通道大小，默认为10', default=10)
    parser.add_argument('--conv2nd_out', type=int, help='指定第二个卷积层的输出通道大小，默认为20', default=20)
    parser.add_argument('--conv2nd_size', type=int, help='指定第二个卷积层的卷积核大小，默认为5', default=5)
    parser.add_argument('--pooling_size', type=int, help='指定池化层的步长，默认为2', default=2)
    parser.add_argument('--Linear_in', type=int, help='指定线性层的输入维度，默认为320', default=320)
    parser.add_argument('--Linear_out', type=int, help='指定线性层的输出维度，默认为10', default=10)
    parser.add_argument('--batch_size', type=int, help='指定训练集和测试集每个batch的的样本数目，默认为64', default=64)
    return parser

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        parser = get_parser()
        args = parser.parse_args()
        self.conv1 = torch.nn.Conv2d(args.conv1st_in, args.conv1st_out, args.conv1st_size)  # 输入通道为1，输出通道为10
        self.conv2 = torch.nn.Conv2d(args.conv2nd_in, args.conv2nd_out, args.conv2nd_size)  # 输入通道为10，输出通道为20
        self.pooling = torch.nn.MaxPool2d(args.pooling_size)  # 最大池化层核为2*2
        self.fc = torch.nn.Linear(args.Linear_in, args.Linear_out)  # 线性层，输入维度为320，输出维度为10

    def forward(self, x):
        # batch_size2存放照片的数目，x为张量，取此张量第一个维度的数目，便是照片数目
        batch_size2 = x.size(0)
        x = self.conv1(x)  # 经过一个卷积层
        x = self.pooling(x)  # 经过一个最大池化层
        x = F.relu(x)  # 经过非线性层
        x = self.conv2(x)  # 经过一个卷积层
        x = self.pooling(x)  # 经过一个最大池化层
        x = F.relu(x)  # 经过非线性层
        # view将其变成全连接网络所需要的输入
        x = x.view(batch_size2, -1)
        x = self.fc(x)
        return x

model=torch.load('mnist_net.pt')
model.eval()

# 这里使用交叉熵损失
criterion = torch.nn.CrossEntropyLoss()
# 由于模型比较大，所以引入冲量来优化训练过程
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 把一轮循环封装到函数中
def train(epoch):
    running_loss = 0.0
    # 取出训练样本
    # batch_idx从0到937
    for batch_idx, data in enumerate(train_loader, 0):
        # x与y
        inputs, target = data

        # 优化器清零
        optimizer.zero_grad()

        # 前馈+反馈+更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # 求累计的loss
        running_loss += loss.item()
        # 每937轮输出一次
        if batch_idx == 937:
            print('[epoch = %d] loss:%.3f' % (epoch + 1, running_loss / 937))
            loss_list.append(running_loss / 937)
            running_loss = 0.0


# 测试中不需要梯度
def test():
    # 正确数目
    correct = 0
    # 总数
    total = 0
    # 包裹在no_grad()中的代码不会进行梯度计算，也不会进行反向传播
    with torch.no_grad():
        # 从test_loader中拿数据
        for data in test_loader:
            images, labels = data
            # 做预测，每个输出都是一个矩阵，每一行有十个量，要找到其中最大值的下标

            outputs = model(images)
            # 于是就要用到max函数，沿着第一个维度（第0个维度是竖着来，第1个维度是横着来）去找最大值的下标，
            # 返回的值，第一个是最大值的值，第二个值是下标
            _, predicted = torch.max(outputs.data, dim=1)
            # total为总数 labels是个矩阵 其中表示N个样本分别为第几类 是N*1的矩阵，取它的第0个元素 就是N
            total += labels.size(0)
            # 相等就为1 不相等就是0 算算它总的正确数目
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))
    accuracy_list.append(correct / total)


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    for epoch in range(1):
        epoch_list.append(epoch)
        train(epoch)
        test()
    plt.plot(epoch_list, accuracy_list, color='r', label='accuary')
    plt.plot(epoch_list, loss_list, color=(0, 0, 0), label='loss')
    plt.xlabel('epoch')
    plt.legend()  # 每条折线的label显示
    plt.show()
