import os
import gzip
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset, dataloader


def get_parser():
    """
    参数初始化
    :return:parser
    """
    parser = argparse.ArgumentParser(description='Tuning')
    parser.add_argument('--n_epochs', default=30, type=int)
    parser.add_argument('--batch_size_train', default=64, type=int)
    parser.add_argument('--batch_size_test', default=1000, type=int)
    parser.add_argument('--initial_lr', default=0.01, type=int)
    parser.add_argument('--momentum', default=0.9, type=int)
    return parser


def load_data(data_folder, data_name, label_name):
    """
    data_folder: 文件目录
    data_name： 数据文件名
    label_name：标签数据文件名
    """
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    return x_train, y_train


class DealDataset(Dataset):
    """
    读取数据、初始化数据
    """
    def __init__(self, folder, data_name, label_name, transform=None):
        train_set, train_labels = load_data(folder, data_name, label_name)
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

# class Net(nn.Module):
#     """
#     构建模型
#     效果太差：87% 过拟合
#     """
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=3)  # 输入通道数为1，输出通道数为10
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=3)  # 输入通道数为10，输出通道数为20
#         self.conv3 = nn.Conv2d(20, 30, kernel_size=3)  # 输入通道数为20，输出通道数为30
#         self.fc1 = nn.Linear(30, 10)  # 全连接
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 10个3*3的卷积核，28*28*1-->26*26*10-->2x2窗口的最大池化-->13*13*10
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 20个3*3的卷积核，卷积核的规模3*3*10*20，13*13*10-->11*11*20-->5*5*20
#         x = F.relu(F.max_pool2d(self.conv3(x), 2)) # 30个3*3的卷积核，卷积核的规模3*3*10*20*30，5*5*20-->3*3*30-->1*1*30
#         x = x.view(-1, 30)  # 平铺 1*1*30 = 30
#         x = F.relu(self.fc1(x))  # 第一次全连接，1*30，30*10 = 1*10
#         return F.log_softmax(x)  # 在softmax的结果上再做多一次log运算

class Net(nn.Module):
    """
    构建模型
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 输入通道数为1，输出通道数为10
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 输入通道数为10，输出通道数为20
        # self.conv2_drop = nn.Dropout2d()  # 我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征
        self.fc1 = nn.Linear(320, 10)  # 全连接
        # self.fc2 = nn.Linear(64, 10) #  本来尝试两次全连接，效果不好，一次全连接准确率由97%-->98%

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 8个5*5的卷积核，卷积核的滑动步长为1（默认），28*28*1-->24*24*10-->2x2窗口的最大池化-->12*12*10
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 20个5*5的卷积核，卷积核的规模5*5*10*20，12*12*10-->8*8*20-->4*4*20
        x = x.view(-1, 320)  # 平铺 4*4*20 = 320
        x = self.fc1(x)  # 第一次全连接，1*320，320*10 = 1*10
        return F.log_softmax(x, dim=1)  # 在softmax的结果上再做多一次log运算


# class Net(nn.Module):
#     """
#     构建模型
#     """
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 8, kernel_size=5)  # 输入通道数为1，输出通道数为8
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=5)  # 输入通道数为8，输出通道数为16
#         # self.conv2_drop = nn.Dropout2d()  # 我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征
#         self.fc1 = nn.Linear(256, 10)  # 全连接
#         # self.fc2 = nn.Linear(64, 10) #  本来尝试两次全连接，效果不好，一次全连接准确率由97%-->98%
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 8个5*5的卷积核，卷积核的滑动步长为1（默认），28*28*1-->24*24*8-->2x2窗口的最大池化-->12*12*8
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 20个5*5的卷积核，卷积核的规模5*5*8*16，12*12*8-->8*8*16-->4*4*16
#         x = x.view(-1, 256)  # 平铺 4*4*16 = 256
#         x = self.fc1(x)  # 第一次全连接，1*256，256*10 = 1*10
#         return F.log_softmax(x, dim=1)  # 在softmax的结果上再做多一次log运算

def train(network, train_loader, epoch, optimizer, log_interval, train_losses):
    """
    模型训练
    :param network: cnn模型
    :param train_loader: 训练数据集
    :param epoch: 轮次
    :param optimizer: 优化器
    :param log_interval: 日志间隔
    :param train_losses: 每一轮的平均损失
    :param optimal_loss: 最优损失
    :return: optimal_loss
    """
    network.train()  # 训练模式network.eval()评估模式
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度置0 更新梯度
        output = network(data)  # 前向传播求出预测值
        loss = F.nll_loss(output, target)  # CrossEntropyLoss()=log_softmax() + NLLLoss() 交叉熵 loss值是整个batch的平均loss
        loss_sum += loss.item()
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 更新所有参数
        if batch_idx % log_interval == 0:  # 每10次更新日志
            print('Train Epoch: {}\tTrained Data:{}/{} ({:.1f}%)\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    train_losses.append(loss_sum / batch_idx)  # 计算一个epoch中每一个样本的loss值
    return loss_sum / batch_idx


def test(network, test_loader, test_losses, accuracies, f1_scores):
    """
    模型测试
    :param network: cnn模型
    :param test_loader: 测试数据
    :param test_losses: 每一轮的平均损失
    :param accuracies: 准确率
    :param f1_scores: f1=2*r*p/(r+p)
    :return: None
    """
    network.eval()
    test_loss = 0  # 记录平均test_loss
    correct = 0  # 记录正确数
    pred_all = []
    label_all = []
    with torch.no_grad():  # 每次要将梯度清零 不要利用BP算法反向传播
        for data, target in test_loader:
            output = network(data)  # 前向传播求出预测值
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累计loss
            pred = output.data.max(1, keepdim=True)[1]  # 输出最大预测率的预测值
            correct += pred.eq(target.data.view_as(pred)).sum()  # 比对 .sum()是为了计算整个batch的总正确数
            pred_all.extend(pred.numpy())  # 将tensor转成numpy
            label_all.extend(target.numpy())

    test_loss /= len(test_loader.dataset)  # 计算每一个样本的平均loss
    test_losses.append(test_loss)
    print('\nTest: Average_loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    accuracies.append(correct / len(test_loader.dataset))  # 保存准确率
    f1_scores.append(f1_score(label_all, pred_all, average='macro'))  # 保存f1
    print('Confusion Matrix\n', confusion_matrix(label_all, pred_all))


def main():

    """
    参数定义
    """
    arg = get_parser().parse_args()

    n_epochs = arg.n_epochs  # 循环整个训练数据集的次数
    batch_size_train = arg.batch_size_train  # 每个训练batch大小
    batch_size_test = arg.batch_size_test  # 每个测试batch大小
    initial_lr = arg.initial_lr  # 初始学习率
    momentum = arg.momentum  # 动量 若选择0.2，准确率会下降0.3个百分比，使用动量减少了收敛过程中的振荡，提高了精度
    log_interval = 10  # 日志记录间隔
    random_seed = 2  # 随机种子
    torch.manual_seed(random_seed)  # 固定随机初始化的权重值
    optimal_loss = 10  # 最优损失

    # """
    # 加载数据
    # """
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(
    #         (0.1307,), (0.3081,))
    # ])
    #
    # trainset = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    # testset = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transform)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True)

    """
    加载数据
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 0.1307和0.3081是MNIST数据集的全局平均值和标准偏差
    ])

    # 实例化DealDataset类
    trainDataset = DealDataset('data/MNIST/raw', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                               transform=transform)
    testDataset = DealDataset('data/MNIST/raw', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
                              transform=transform)

    # 训练数据和测试数据的装载
    train_loader = dataloader.DataLoader(
        dataset=trainDataset,
        batch_size=batch_size_train,  # 一个批次可以认为是一个包，每个包中含有100张图片
        shuffle=True,
    )

    test_loader = dataloader.DataLoader(
        dataset=testDataset,
        batch_size=batch_size_test,
        shuffle=True,
    )

    """
    构建网络
    """
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=initial_lr, momentum=momentum)  # 实现随机梯度下降（优化器）
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 加上动态学习率后，准确率：99%->99.2%

    """
    模型训练与测试
    """
    train_losses = []  # 保存训练每一轮的平均loss
    train_counter = [i for i in range(1, n_epochs + 1)]
    test_losses = []  # 保存测试每一轮的平均loss
    test_counter = [i for i in range(1, n_epochs + 1)]
    accuracies = []  # 精度
    f1_scores = []

    for epoch in range(1, n_epochs + 1):  # 循环3轮->循环30轮->循环50轮    精度：98%->99.2%
        loss_ = train(network, train_loader, epoch, optimizer, log_interval, train_losses)
        if loss_ < optimal_loss:
            torch.save(network.state_dict(), './model.pth')  # 保存模型 存放训练过程中需要学习的权重和偏执系数
            torch.save(optimizer.state_dict(), './optimizer.pth')  # 保存优化器
            optimal_loss = loss_
        test(network, test_loader, test_losses, accuracies, f1_scores)
        scheduler.step()

    """
    数据可视化
    """
    fig1 = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.plot(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Epoch_number')
    plt.ylabel('loss')
    plt.show()

    fig2 = plt.figure()
    plt.plot(test_counter, accuracies, color='blue')
    plt.xlabel('Epoch_number')
    plt.ylabel('Accuracy')
    plt.show()

    fig3 = plt.figure()
    plt.plot(test_counter, f1_scores, color='blue')
    plt.xlabel('Epoch_number')
    plt.ylabel('f1_score')
    plt.show()


# python Picture_Recognition.py --n_epochs 50
if __name__ == '__main__':
    main()