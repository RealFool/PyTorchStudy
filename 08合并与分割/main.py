import torch

'''
cat
torch.cat()是为了把多个tensor进行拼接而存在的
函数目的： 在给定维度上对输入的张量序列seq 进行连接操作。

outputs = torch.cat(inputs, dim=?) → Tensor
    inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
    dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列。

输入数据必须是序列，序列中dim=?之外的数据是任意相同的shape的同类型tensor
维度不可以超过输入数据的任一个张量的维度
'''
print('-------------------cat-----------------------')
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)

print(torch.cat([a, b], dim=0).shape)

print('-------------------For example-----------------------')
a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)

print(torch.cat([a1, a2], dim=0).shape)

a2 = torch.rand(4, 1, 32, 32)
# print(torch.cat([a1, a2], dim=0).shape)  # Sizes of tensors must match except in dimension 0. Expected size 3 but got size 1 for tensor number 1 in the list.

print(torch.cat([a1, a2], dim=1).shape)

a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)

print(torch.cat([a1, a2], dim=2).shape)

'''
stack
create new dim

函数的意义：使用stack可以保留两个信息：[1. 序列] 和 [2. 张量矩阵] 信息，属于【扩张再拼接】的函数。
形象的理解：假如数据都是二维矩阵(平面)，它可以把这些一个个平面按第三维(例如：时间序列)压成一个三维的立方体，而立方体的长度就是时间序列长度。
该函数常出现在自然语言处理（NLP）和图像卷积神经网络(CV)中。

stack()
官方解释：沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
浅显说法：把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。

outputs = torch.stack(inputs, dim=?) → Tensor
参数
    inputs : 待连接的张量序列。
    注：python的序列数据只有list和tuple。
    
    dim : 新的维度， 必须在0到len(outputs)之间。
    注：len(outputs)是生成数据的维度大小，也就是outputs的维度值。
    
函数中的输入inputs只允许是序列；且序列内部的张量元素，必须shape相等
----举例：[tensor_1, tensor_2,..]或者(tensor_1, tensor_2,..)，且必须tensor_1.shape == tensor_2.shape

dim是选择生成的维度，必须满足0<=dim<len(outputs)；len(outputs)是输出后的tensor的维度大小
'''
print('-------------------stack-----------------------')
a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)

print(torch.stack([a1, a2], dim=2).shape)

a = torch.rand(32, 8)
b = torch.rand(32, 8)

print(torch.stack([a, b], dim=0).shape)

'''
cat v.s. stack
'''
print('-------------------cat v.s. stack-----------------------')
a = torch.rand(32, 8)
b = torch.rand(30, 8)

# stack expects each tensor to be equal size, but got [32, 8] at entry 0 and [30, 8] at entry 1
# print(torch.stack([a, b], dim=0).shape)

print(torch.cat([a, b], dim=0).shape)

'''
Split: by len

 torch.split()作用将tensor分成块结构。
'''
print('-------------------Split: by len-----------------------')
a = torch.rand(32, 8)
b = torch.rand(32, 8)

c = torch.stack([a, b], dim=0)
print(c.shape)

# split_size_or_sections为list型时，按照list中的num分割对应的dim
# 按照dim=0这个维度去分，每大块包含1个小块
aa, bb = c.split([1, 1], dim=0)
print(aa.shape, bb.shape)

# 按照dim=0这个维度去分，每大块分别为[10, 10, 12]个小块
aa, bb, cc = c.split([10, 10, 12], dim=1)
print(aa.shape, bb.shape, cc.shape)

# 按照dim=0这个维度去分，每大块包含1个小块
aa, bb = c.split(1, dim=0)
print(aa.shape, bb.shape)

# 按照dim=1这个维度去分，每大块包含30个小块
aa, bb = c.split(30, dim=1)
print(aa.shape, bb.shape)

'''
Chunk: by num

chunk 与 split的区别

（1）chunks只能是int型，而split_size_or_section可以是list。

（2）chunks在时，不满足该维度下的整除关系，会将块按照维度切分成1的结构。而split会报错。

'''
print('-------------------Chunk: by num-----------------------')
a = torch.rand(32, 8)
b = torch.rand(32, 8)

c = torch.stack([a, b], dim=0)
print(c.shape)

# plit,不满足该维度下的整除关系，会报错，而chunk不会
# not enough values to unpack (expected 2, got 1)
# aa, bb = c.split(2, dim=0)

# chunks在时，不满足该维度下的整除关系，会将块按照维度切分成1的结构
aa, bb = c.chunk(2, dim=0)
print(aa.shape, bb.shape)
