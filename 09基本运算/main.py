import torch

'''
basic
+ - * /
'''
print('-------------------basic-----------------------')
a = torch.rand(3, 4)
b = torch.rand(4)
print(a)
print(b)

print(a + b)
print(torch.add(a, b))

print(torch.all(torch.eq(a-b, torch.sub(a, b))))
print(torch.all(torch.eq(a*b, torch.mul(a, b))))
print(torch.all(torch.eq(a/b, torch.div(a, b))))

'''
matmul
矩阵乘法
mm, matmul, @

torch.matmul()的用法比torch.mm更高级，torch.mm只适用于二维矩阵，而torch.matmul可以适用于高维。当然，对于二维的效果等同于torch.mm()
@ 等同于 matmul

'''
print('-------------------matmul-----------------------')
# tensor([[3., 3.],
#         [3., 3.]])
# torch.full()数据类型会根据给定的fill_value推断出来
a = torch.full([2, 2], 3.0)
print(a)

b = torch.ones(2, 2)
print(b)

print(torch.mm(a, b))
print(torch.matmul(a, b))
print(a@b)

'''
An example
'''
print('-------------------An example-----------------------')
x = torch.rand(4, 784)
w = torch.rand(512, 784)

print((x @ w.t()).shape)

a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
print(torch.matmul(a, b).shape)

# Broadcast自动扩展
b = torch.rand(4, 1, 64, 32)
print(torch.matmul(a, b).shape)

'''
Power

pow():平方
sqrt()():取平方根
rsqrt():对每个元素取平方根后再取倒数
'''
print('-------------------Power-----------------------')
a = torch.full([2, 2], 3)
print(a)
print(a.pow(2))

aa = a**2
print(aa)
print(aa.sqrt())
print(aa**0.5)

print(aa.rsqrt())

'''
Exp log
torch.exp(x)，e的x次幂
torch.log(x)，以e为底，x的对数
'''
print('-------------------Exp log-----------------------')
print(torch.e)
a = torch.exp(torch.ones(2, 2))
print(a)

print(torch.log(a))

'''
Approximation  近似值

.floor() ：向下取整
.ceil()  ：向上取整
.round() ：近似，将输入input张量的每个元素舍入到最近的整数。
.trunc() ：取整数部分
.frac()  ：取小数部分
'''
print('-------------------Approximation-----------------------')
a = torch.tensor(3.14)
print(a)

print(torch.floor(a))
print(torch.ceil(a))
print(torch.trunc(a))
print(torch.frac(a))

print(torch.round(a))
a = torch.tensor(3.5)
print(torch.round(a))

'''
clamp 夹，函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
median(): 返回所有元素的中位数,如果共有偶数个元素，则会有两个中位数，返回较小的那一个。

'''
print('-------------------clamp-----------------------')
grad = torch.rand(2, 3)*15
print(grad)
print(torch.max(grad))
print(torch.median(grad))

print(grad.clamp(10))
print(grad.clamp(10, 12))
print(torch.clamp(grad, 10, 12))
