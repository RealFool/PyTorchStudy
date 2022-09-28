import torch

'''
Pytorch-统计学
更多知识可参考：https://zhuanlan.zhihu.com/p/479933987
'''

'''
范数
常用的范数有：
L1范数，也叫曼哈顿距离：是一个向量中所有元素的绝对值之和。
L2范数，也叫欧几里得范数：是一个向量中所有元素取平方和，然后再开平方。

torch.norm是对输入的Tensor求范数
参数：
    input (Tensor) – 输入张量
    p (float) – 范数计算中的幂指数值，默认为2
    dim (int) – 缩减的维度
    out (Tensor, optional) – 结果张量
    keepdim（bool）– 保持输出的维度  （此参数官方文档中未给出，但是很常用）
'''
print('-------------------norm-----------------------')
a = torch.full([8], 1.0)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print(b)
print(c)

print(torch.norm(a, p=1), torch.norm(b, p=1), torch.norm(c, p=1))
print(torch.norm(a), torch.norm(b), torch.norm(c))  # 默认p=2

# 按1维度求1范数
print(torch.norm(b, p=1, dim=1))
# 按1维度求2范数
print(torch.norm(b, p=2, dim=1))

'''
mean, sum, min, max, prod

prod
    torch.prod(input, dim, keepdim=False, *, dtype=None) → Tensor
    返回输入张量给定维度上每行的积。 默认所有维度的乘积

'''
print('-------------------mean, sum, min, max, prod-----------------------')
a = torch.arange(8).view(2, 4).float()
print(a)

print(torch.min(a), torch.max(a), torch.mean(a))
print(torch.prod(a))
print(torch.prod(a, dim=1))

print(torch.sum(a))

# argmax函数：torch.argmax(input, dim=None, keepdim=False) 返回指定维度最大值的序号
print(torch.argmax(a), torch.argmin(a))
print(torch.argmax(a, dim=0), torch.argmin(a, dim=1))
'''
argmin, argmax

argmax
    argmax函数：torch.argmax(input, dim=None, keepdim=False) 返回指定维度最大值的序号
    可指定维度，默认会把tensor拉平返回索引
'''
print('-------------------argmin, argmax-----------------------')

a = a.view(1, 2, 4)
print(a)
print(torch.argmax(a))
print(torch.argmin(a))

a = torch.rand(2, 3, 4)
print(a)
print(torch.argmax(a))

a = torch.rand(4, 10)
print(a)
print(a[0])
print(torch.argmax(a))
print(torch.argmax(a, dim=1))

'''
dim, keepdim

keepdim（bool）– 保持输出的维度
'''
print('-------------------dim, keepdim-----------------------')
print(a)

print(torch.max(a, dim=1))  # 指定维度1
print(torch.argmax(a, dim=1))

print(torch.max(a, dim=1, keepdim=True))
print(torch.argmax(a, dim=1, keepdim=True))

'''
Top-k or k-th

topk() 返回 列表中最大的n个值
    input -> 输入tensor
    k -> 前k个
    dim -> 默认为输入tensor的最后一个维度
    sorted -> 是否排序
    largest -> False表示返回第k个最小值
    
y, i = torch.kthvalue(x, k, n) 沿着n维度返回第k小的数据。

'''
print('-------------------Top-k or k-th-----------------------')
print(a)
print(torch.topk(a, 3, dim=1))
print(torch.topk(a, 3, dim=1, largest=False))

# 沿着1维度返回第8小的数据
print(torch.kthvalue(a, 8, dim=1))
print(torch.kthvalue(a, 3))  # 默认最后一个维度
print(torch.kthvalue(a, 3, dim=1))

'''
compare
    ▪ >, >=, <, <=, !=, ==
    ▪ torch.eq(a, b)
        ▪ torch.equal(a, b)

'''
print('-------------------compare-----------------------')
print(a > 0)

print(torch.gt(a, 0))

print(torch.gt(a, 0.5))

print(a != 0)

a = torch.ones(2, 3)
b = torch.randn(2, 3)
print(a)
print(b)
print(torch.eq(a, b))
print(torch.eq(a, a))
