import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn
import numpy as np


img_tran = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root="./dataset", train=True, transform=img_tran, download=True)
test_set = torchvision.datasets.MNIST(root="./dataset", train=False, transform=img_tran, download=True)

print(len(train_set))
print(test_set.classes)
img, target = train_set[0]
print(img.shape)
# img.show()



# writer = SummaryWriter('datalodaer')
# step = 0
# for data in train_loader:
#     imgs, targets = data
#     # print(imgs.shape)
#     # print(targets)
#     writer.add_images("train_data", imgs, step)
#     step += 1
#
# writer.close()


# 残差块
class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.stride = stride
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu1 = torch.nn.ReLU(inplace=True)  # 原地操作 复用一个
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        # 输入输出通道数不同，通过1*1卷积改变数据的通道数
        if in_channels != out_channels:
            self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        out1 = self.relu1(self.bn1(self.conv1(x)))  # a(l+1)
        out2 = self.bn2(self.conv2(out1))  # z(l+2)

        if self.conv1x1:
            x = self.conv1x1(x)       # a(l)
        out = self.relu1(out2 + x)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels=64, kernel_size=(7,7), stride=2, padding=3),  # 64 14 14
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 64 7 7
            Residual(64, 64),
            Residual(64, 64),
            Residual(64, 64)
        )

        self.conv3 = torch.nn.Sequential(
            Residual(64, 128, stride=2),  # 128 4 4
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128)
        )

        self.conv4 = torch.nn.Sequential(  # 256 2 2
            Residual(128, 256,stride=2),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
        )

        self.conv5 = torch.nn.Sequential(
            Residual(256, 512, stride=2),
            Residual(512, 512),
            Residual(512, 512)
        )

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # 每个通道大小为（1,1）
        self.fc = torch.nn.Linear(512, num_classes)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avg_pool(out)
        out = out.view(out.size()[0], -1)

        out = self.fc(out)
        return out

    def test(self, x):
        pred = self.forward(x)
        pred = self.softmax(pred)
        return pred


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

num_epoch = 50
learning_rate = 0.01
minbatch_size = 32
costs = []



# X_train, X_test, Y_train, Y_test, classes = load_data()
# X_train = X_train.to(device)
# Y_train = Y_train.to(device)
# X_test = X_test.to(device)
# Y_test = Y_test.to(device)

m = ResNet(1, num_classes=10)
m = m.to(device)  # 初始化网络

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True, num_workers=0, drop_last=False)

for epoch in range(num_epoch):
    cost = 0
    for i, data in enumerate(train_loader):
        X_train, Y_train = data
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)

        acc_train = 0
        optimizer.zero_grad()
        y_pred = m.forward(X_train)
        loss = loss_fn(y_pred, Y_train.long())
        loss.backward()
        optimizer.step()

        trainpred = m.test(X_train)
        trainpred = torch.argmax(trainpred, dim=1)
        acc_train = ((trainpred == Y_train) / X_train.size()[0]).sum().float()
        acc_train = acc_train.cpu().detach().numpy()
        cost = cost + loss.cpu().detach().numpy()
    costs.append(cost / (i + 1))

    for i, data in enumerate(test_loader):
        acc_test = 0
        X_test, Y_test = data
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        testpred = m.test(X_test)
        testpred = torch.argmax(testpred, dim=1)
        acc_test = ((testpred == Y_test)/X_test.size()[0]).sum().float()

    if epoch % 1 == 0:
        print("epoch = " + str(epoch) + ":  loss" + str(cost/(i+1)))
        print("训练集准确率" + str(acc_train))
        print("测试集准确率" + str(acc_test))

