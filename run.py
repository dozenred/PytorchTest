import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import configparser
import visdom
from sklearn.metrics import f1_score


class Minist:
    # 卷积神经网络定义
    class Net(torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)

            # 2x2池化
            self.pooling = torch.nn.MaxPool2d(2)

            # 卷积层
            self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
            # 1大小卷积层，虽然加快了速度，但是损失了精度
            # self.conv1x1 = torch.nn.Conv2d(10, 20, kernel_size=1)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)

            # 全连接层
            self.fc = torch.nn.Linear(512, 10)

        def forward(self, x):

            batch_size = x.size(0)

            # 转化到二维张量
            # x = x.view(-1, 784)
            x = F.relu(self.pooling(self.conv1(x)))
            # x = F.relu(self.conv1x1(x))
            x = F.relu(self.pooling(self.conv2(x)))
            # 展开
            x = x.view(batch_size, -1)
            # 全连接
            x = self.fc(x)
            return x

    # 初始化
    def __init__(self, paraPath="", modelpath="") -> None:
        # 加载配置
        self.config = configparser.ConfigParser()
        self.paraPath = paraPath
        self.modelpath = modelpath
        self.loadParams(paraPath)

        # 配置可视化
        # 将窗口类实例化
        self.isVisual = self.getParam(
            "env", "isVisualization", False) == "True"
        if self.isVisual:
            self.vis = visdom.Visdom(env='MINIST')

        # 定义训练模型
        self.model = self.Net()

        isLoadModel = self.getParam('env', 'idLoadModel', False) == "True"
        if modelpath != "" and isLoadModel:
            savedModel = self.loadModel(modelpath)
            self.epoch = savedModel['epoch']    # 加载epoch
            self.model.load_state_dict(savedModel['state_dict'])
        else:
            self.epoch = 0

        # 定义设备
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available()
            and self.getParam('env', 'isUseGPU', False) == "True" else "cpu")
        self.model.to(self.device)

        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()
        # 优化器
        lr = float(self.getParam("model", "l1", 0.01))
        momentum = float(self.getParam("model", "momentum", 0.5))
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum)

        if modelpath != "" and isLoadModel:
            self.optimizer.load_state_dict(savedModel['optimizer'])  # 加载epoch

    # 训练
    def train(self, epoch):
        self.model.train()
        running_loss = 0.0
        total_loss = 0.0
        print(f'Epoch {epoch+1}')
        batch_idx = 0
        for batch_idx, (inputs, target) in enumerate(self.train_loader):
            # 数据迁移到同一块显卡
            inputs, target = inputs.to(self.device), target.to(self.device)

            # 优化器清零
            self.optimizer.zero_grad()
            # 训练
            outputs = self.model(inputs)
            # 交叉熵
            loss = self.criterion(outputs, target)
            # 反向传播
            loss.backward()
            # 更新
            self.optimizer.step()

            running_loss += loss.item()
            if batch_idx % 300 == 299:
                print(f'{batch_idx+1} loss: {running_loss /300}')
                total_loss += running_loss
                running_loss = 0.0
        # 更新窗口图像
        if self.isVisual:
            self.vis.line([total_loss/(batch_idx+1)], [epoch],
                          win='train_loss', update='append')
        self.epoch = epoch

    # 测试
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        # 预测与真实值
        predicteds = []
        trueClass = []
        CMT = torch.zeros(10, 10, dtype=torch.int64)
        # 不开启计算梯度
        with torch.no_grad():
            for images, labels in self.test_loader:
                # 数据迁移到同一块显卡
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                # 获取识别的结果
                _, predicted = torch.max(outputs.data, dim=1)

                # 比较识别正确的结果
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                for i in range(labels.size(0)):
                    CMT[predicted[i], labels[i]] += 1

                # 收集识别数据和
                predicteds.extend(predicted.cuda().data.cpu().numpy())
                trueClass.extend(labels.cuda().data.cpu().numpy())

            accutacy = 100*correct / total
            print(f'Accutacy on test set: {accutacy}')
            print(CMT.numpy())  # 混淆矩阵

            f1 = f1_score(trueClass, predicteds,
                          average="macro")  # f1
            print('----------------------------------------------------\n')

            if self.isVisual:
                self.vis.line([f1], [self.epoch+1],
                              win='f1_score', update='append')
                self.vis.line([accutacy], [self.epoch+1],
                              win='accutacy', update='append')

    # 运行
    def run(self):
        # 读取epoch的次数
        epochs = int(self.getParam("model", "epoch", 10))
        b = self.epoch
        # 创建窗口并初始化
        if self.isVisual:
            self.vis.line([0.], [0], win='train_loss',
                          opts=dict(title='Train Loss'))
            self.vis.line([0.], [0], win='accutacy',
                          opts=dict(title='Accutacy'))
            self.vis.line([0.], [0], win='f1_score',
                          opts=dict(title='f1 Score'))
        for epoch in range(b, b + epochs):
            self.train(epoch)
            self.test()

    # 加载数据
    def loadData(self, path):
        # 转变成张量,同时映射到[0, 1]
        # PIL (w*h*c) -> (c*w*h)
        # Compose串联多个变换操作
        transform = transforms.Compose([
            transforms.ToTensor(),                       # 转变成张量
            transforms.Normalize((0.1307,), (0.3081,))   # 标准化
        ])

        # 导入数据
        # (数据保存的目录, 提取训练集吗(False则提取测试集), 没有的时候是否去下载数据)
        # TODO 数据读取
        # self.train_set = datasets.MNIST(
        #     root=path, train=True, transform=transform)
        # self.test_set = datasets.MNIST(
        #     root=path, train=False, transform=transform)

        # 数据加载
        # (数据集，batch的个数)
        self.train_loader = DataLoader(
            dataset=self.train_set, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(
            dataset=self.test_set, batch_size=32, shuffle=False)

    # 加载模型参数
    def loadModel(self, path):
        return self.model.load_state_dict(torch.load(path))

    # 加载配置
    def loadParams(self, path=""):
        if path != "":
            self.config.read(path, encoding='utf-8')

    def getParam(self, section, options, defualt):
        if(self.config.has_option(section, options)):
            return self.config.get(section, options)
        else:
            self.config.set(section, options, str(defualt))
            return defualt

    # 保存模型
    def saveModel(self, path):
        # 保存计算参数
        torch.save({'epoch': self.epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   path)

    # 保存参数
    def saveParams(self):
        with open(self.paraPath
                  if self.paraPath != "" else "./config/config.ini", 'w') as f:
            self.config.write(f)


if __name__ == '__main__':
    # 读配置文件并初始化训练
    m = Minist("./config/config.ini", "./save/checkpoint.pth")
    # 加载数据
    m.loadData('./data')
    m.run()
    # 保存模型
    m.saveModel()
