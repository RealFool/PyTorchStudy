import torch
import numpy as np

'''
torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, 
返回一个符合均值为0，方差为1的正态分布（标准正态分布）中填充随机数的张量
size(int…) --定义输出张量形状的整数序列。可以是数量可变的参数，也可以是列表或元组之类的集合
'''
print('--------------------randn()----------------------')
a = torch.randn(2, 3)
print(a)
print(a.type())

print(type(a))

'''
isinstance()判断一个函数是否为已知的类型
'''
print('--------------------isinstance()----------------------')
print(isinstance(a, torch.FloatTensor))

# print(isinstance(a, torch.cuda.DoubleTensor))
#
# data = a.cuda()
#
# print(isinstance(data, torch.cuda.DoubleTensor))


'''
torch.Tensor是存储和变换数据的主要工具。
Tensor与Numpy的多维数组非常相似。
Tensor还提供了GPU计算和自动求梯度等更多功能，这些使Tensor更适合深度学习。
更多内容可见本文：https://blog.csdn.net/sazass/article/details/109304327

torch.tensor()根据数据直接创建
'''
print('--------------------Dim 0----------------------')
'''
Dim 0的经常用作计算loss
'''
print(torch.tensor(1.))
print(torch.tensor(1.300))

a = torch.tensor(2.2)
# size/shape都指形状
print(a.shape)
print(len(a.shape))
print(a.size())

print('-------------------Dim 1-----------------------')
'''
Dim 1的一般用在Bias或者Linear Input
'''
print(torch.tensor(([1.1])))
print(torch.tensor([1.1, 2.2]))

# 指定维度，随机生成
print(torch.FloatTensor(1))
print(torch.FloatTensor(2))

# numpy转tensor
data = np.ones(2)
print(data, type(data))
print(torch.from_numpy(data))

print('-------------------Dim 2-----------------------')
'''
经常适用于Linear Input batch

例如：假设一张图片用 Dim 1 的[784]向量来表达
此时若要一次性输入多张图片，我们可以多加一个维度用来表示图片个数，比如四张图片[4, 784]
此时，这个tensor的维度就是2，size就是[4, 784]
'''
# 定义一个Dim 2的
a = torch.randn(2, 3)
print(a)
print(a.shape, a.size())
# 可以用索引获取每个维度的数据个数
print(a.size(0))
print(a.size(1))
print(a.shape[1])

print('-------------------Dim 3-----------------------')
'''
常适用于RNN Input Batch
RNN（Recurrent Neural Network）是一类用于处理序列数据的神经网络。
例如：
假设一句话10个单词，我们第一维度表示一句话单词个数，第二维度表示一批数据中句子的个数，第三维度表示每个单词的所有特征
如[10, 20, 100]

torch.rand：返回一个张量，包含了从区间[0,1)的均匀分布中抽取一组随机数，形状由可变参数size定义
'''
a = torch.rand(1, 2, 3)
print(a)
print(a.shape)
print(a[0])
print(list(a.shape))

print('-------------------Dim 4-----------------------')
'''
常适用于图片类型[b, c, h, w]
b: 图片个数
c: 通道个数，黑白图片通道个数为1，彩色图片通道个数为3
h, w: 图片像素高和宽

卷积神经网络（Convolutional Neural Networks, CNN）
'''

a = torch.rand(2, 3, 28, 28)
print(a)
print(a.shape)

print('-------------------numel-----------------------')
'''
numel是指tensor占用内存的大小
'''
print(a.shape)
print(a.numel())  # 2*3*28*28
print(a.dim())  # len(a.shape)
print(torch.tensor(1).dim())
