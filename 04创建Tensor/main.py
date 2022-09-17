import numpy as np
import torch
import math

'''
Import from numpy
'''
print('-------------------Import from numpy-----------------------')
a = np.array([2, 3.3])

print(torch.from_numpy(a))

a = np.ones([2, 3])
print(torch.from_numpy(a))

'''
Import from list

此处建议，tensor和Tensor的用法习惯：
参数为数值时建议使用小写的tensor，参数为shape时建议使用大写的Tensor，不然容易混淆
'''
print('-------------------Import from list-----------------------')
print(torch.tensor([2., 3.2]))  # 传数值建议使用小写tensor

print(torch.FloatTensor([2., 3.2]))  # 大写Tensor虽然也可以穿数值，但不建议使用，建议传shape时使用它

print(torch.tensor([[2., 3.2], [1., 22.3]]))

'''
uninitialized
使用未初始化的数据往会带来一些问题，例如数据无穷大或无穷小或nan的问题，所以，未初始化数据要及时覆盖
'''
print('-------------------uninitialized-----------------------')
print(torch.empty(1))

print(torch.Tensor(2, 3))

print(torch.IntTensor(2, 3))

print(torch.FloatTensor(2, 3))

'''
set default type

默认一般时FloatTensor
增强学习一般使用double，其他一般使用float
'''
print('-------------------set default type-----------------------')
print(torch.tensor([1.2, 3]).type())
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor(([1.2, 3])).type())

'''
rand/rand_like, randint

rand:返回一个张量，包含了从区间[0,1)的均匀分布中抽取一组随机数，形状由可变参数size定义
rand_like:读取一个张量的shape，然后调用rand返回和该张量相同shape的张量
randint:torch.randint(low, high, size)只能采样整数，[min, max)，shape
'''
print('-------------------rand/rand_like, randint-----------------------')
a = torch.rand(3, 3)
print(a)
print(torch.rand_like(a))
# 均匀采样0~10的tensor
x = 10 * a
print(x)

print(torch.randint(1, 10, [3, 3]))

'''
randn:正态分布，默认均值和方差为N(0, 1)，若想自定义均值和方差，则用到normal
torch.normal(mean=0.,std=1.,size=(2,2))
'''
print('-------------------randn-----------------------')
print(torch.randn(3, 3))

print(torch.normal(1, 1, size=(2, 2)))

'''
torch.full(size, fill_value)  数据类型会根据给定的fill_value填充size规定的shape
'''
print('-------------------full-----------------------')
print(torch.full([2, 3], 7))
print(torch.full([], 7))    # 生成标量
print(torch.full([1], 7))

'''
arange/range
建议使用arange
range将在未来的版本中淘汰，不建议使用
'''
print('-------------------arange/range-----------------------')
print(torch.arange(0, 10))
# print(torch.range(0, 10))

'''
linspace/logspace
linspace(start: Number, end: Number, steps), [start, end]闭区间, steps指生成个数
logspace(start: Number, end: Number, steps, base), [start, end]闭区间, steps指生成个数, base指底数默认为10
'''
print('-------------------linspace/logspace-----------------------')
print(torch.linspace(0, 10, steps=4))
print(torch.linspace(0, 10, steps=10))
print(torch.linspace(0, 10, steps=11))

print(torch.logspace(0, -1, steps=10))
print(math.exp(1))
print(torch.logspace(0, 1, steps=10, base=math.exp(1)))

'''
Ones/zeros/eye
Ones:生成全为1
zeros:生成全为0
eye:生成单位矩阵，若非方阵，则从左侧划分最大方阵为单位阵
'''
print('-------------------Ones/zeros/eye-----------------------')
print(torch.ones(3, 3))

print(torch.zeros(3, 3))

print(torch.eye(3, 3))
print(torch.eye(3, 4))
print(torch.eye(3))

a = torch.zeros(3, 3)
print(torch.ones_like(a))

'''
randperm
数据洗牌， 相当于random.shuffle
'''
print('-------------------randperm-----------------------')
print(torch.randperm(10))
a = torch.rand(2, 3)
b = torch.rand(2, 2)
ids = torch.randperm(2)  # 索引重排
print(ids)

print(a[ids])
print(b[ids])
print(a)
print(b)
