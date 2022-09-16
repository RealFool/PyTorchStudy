import torch
from torch import nn    # nn完成神经网络相关的工作
from torch.nn import functional as F
from torch import optim

import torchvision  # 视觉
from matplotlib import pyplot as plt

from utils import plot_image, plot_curve, one_hot

batch_size = 512    # 一次处理图片的数量

# step1. load dataset
# 加载训练集
train_loader = torch.utils.data.DataLoader(
    # 加载MNIST数据集，'mnist_data/'指下载路径，train=True指定训练集（70K图片，60K用来training，10K用来test）
    # download=True指当前问价若无MNIST数据集，则通过网络下载
    # torchvision.transforms.ToTensor()将下载的numpy格式转化为tensor格式
    # torchvision.transforms.Normalize()正则化数据，因为用神经网络接受的数据最好是在0附近均匀分配，但是我们的数据像素都是大于零的，所以可以减去0.1307并除以标准差0.3081使数据在o附近均匀分布，用于提升性能
    # batch_size=batch_size一次加载多少张图片
    # shuffle=True数据洗牌
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

# 加载测试集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

'''
查看下载的数据集中的数据
python编程中，我们在读文件的时候会经常用到next函数，python3 中next函数可以调用生成器的对象以参数形式传入到next(params)，返回迭代器到下一个项目。
iter() 函数用来返回迭代器对象

x.shape = [512, 1, 28, 28]，表示512个28*28的矩阵，1指一个通道，x[0, 0]则表示第一张图片，第二个0指第一个通道
'''
x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')


# 定义个三层网络
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        # 输入是28*28，即图片的像素，256是自己根据经验定的输出
        self.fc1 = nn.Linear(28 * 28, 256)
        # 这里的输入必须是上一个的输出，即256
        self.fc2 = nn.Linear(256, 64)
        # 同这里的输入是上面的输出，而这里的输出则是个10分类，即0~9
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x


net = Net()
'''
梯度下降算法
net.parameters()返回一组权值[w1, b1, w2, b2, w3, b3]
lr=0.01，指learning rate
momentum=0.9，指动量
'''
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []

# 对数据集迭代三次
for epoch in range(3):
    # 每一次从训练集中取出一个样本，共512张图片
    # enumerate()这个函数的基本应用就是用来遍历一个集合对象，它在遍历的同时还可以得到当前元素的索引位置，在这里，batch_idx即表示索引。
    for batch_idx, (x, y) in enumerate(train_loader):

        '''
            https://blog.csdn.net/TYUT_xiaoming/article/details/100799527
            一般地，在CNN等网络中，都是通过卷积过滤器对目标进行计算，然而这些计算都是建立在高维数据。
            最后，项目需要对数据进行分类或者识别，就需要全连接层Linear，这时候就需要将高维数据平铺变为低位数据。
            在CNN中卷积或者池化之后需要连接全连接层，所以需要把多维度的tensor展平成一维，x.view(x.size(0), -1)就实现的这个功能
            在pytorch中的view()函数就是用来改变tensor的形状的，例如将2行3列的tensor变为1行6列，其中-1表示会自适应的调整剩余的维度
            卷积或者池化之后的tensor的维度为(batchsize，channels，x，y)，其中x.size(0)指batchsize的值，
            最后通过x.view(x.size(0), -1)将tensor的结构转换为了(batchsize, channels*x*y)，即将（channels，x，y）拉直，然后就可以和fc层连接了
        '''
        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]
        x = x.view(x.size(0), 28 * 28)
        # => [b, 10]
        out = net(x)
        # [b, 10]
        '''
            对类别张量进行one-hot编码
            one-hot 形式的编码在深度学习任务中非常常见，但是却并不是一种很自然的数据存储方式。简单来说，就是将类别拆分成一一对应的 0-1 向量
            例个简单的例子：
            把0表示成：0 0 0 0
            把1表示成：0 1 0 0
            把2表示成：0 0 1 0
            把3表示成：0 0 0 1
            实际情况可能更复杂
        '''
        y_onehot = one_hot(y)
        # 损失函数：loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)
        '''
        清零梯度 -> 计算梯度 -> 更新梯度
        '''
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 把梯度更新到权值3当中去，w' = w - lr*grad
        optimizer.step()
        '''
        Returns the value of this tensor as a standard Python number
        Example::
            >>> x = torch.tensor([1.0])
            >>> x.item()
            1.0
        '''
        train_loss.append(loss.item())

        # 每隔10个batch打印一下loss，看看是否降低
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

# loss可视化，loss是衡量training的指标
plot_curve(train_loss)
# # we get optimal [w1, b1, w2, b2, w3, b3]

'''
准确度测试
'''
total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    # out: [b, 10] => pred: [b]
    # 返回最大值的索引，dim=1，指定列，也就是行不变，列之间的比较
    # 关于dim的进一步理解，参考：https://blog.csdn.net/lj2048/article/details/114262597
    pred = out.argmax(dim=1)
    # 这里相当于把y当作掩码，对的索引会返回1，不对的会返回0，再进行求和
    # item()函数，取出张量具体位置的元素元素值，返回的是该位置元素值的高精度值，为了下面的计算
    # 本批数据正确的总个数
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num  # 准确度
print('test acc:', acc)

# 测试样本预测结果可视化
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
