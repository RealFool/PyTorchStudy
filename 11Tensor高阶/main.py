import torch

'''
Tensor高阶操作
'''

'''
where()

torch.where()函数的作用是按照一定的规则合并两个tensor类型。
    torch.where(condition，a，b)
    输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出。
    注意：a和b是tensor.

'''
print('-------------------where-----------------------')
cond = torch.rand([2, 2])
print(cond)
a = torch.zeros([2, 2])
b = torch.ones([2, 2])
# 合并a,b两个tensor，如果cond中元素大于0.5，则output中与a对应的位置取a的值，否则取b的值
print(torch.where(cond > 0.5, a, b))

'''
gather
torch.gather(input, dim, index, out=None) → Tensor
    input (Tensor) – 源张量
    dim (int) – 索引的轴
    index (LongTensor) – 聚合元素的下标
    out (Tensor, optional) – 目标张量
'''
print('-------------------gather-----------------------')
t = torch.Tensor([[1, 2], [3, 4]])
print(torch.gather(t, 1, torch.LongTensor([[0, 0], [1, 0]])))

'''
retrieve label
'''
print('-------------------retrieve label-----------------------')
prod = torch.randn(4, 10)
print(prod)

idx = torch.topk(prod, dim=1, k=3)
print(idx)

idx = idx[1]
print(idx)

label = torch.arange(10) + 100
print(label)    # (1, 10)

print(torch.expand_copy(label, [4, 10]))
print(torch.gather(torch.expand_copy(label, [4, 10]), dim=1, index=idx.long()))
