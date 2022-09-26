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
a = torch.ones(2, 2) * 3
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
