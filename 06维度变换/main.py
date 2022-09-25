import torch

'''
View reshape
0.3版本之前是view()
0.3之后为了与numpy一致，增加了torch.reshape()

变换原则就是保持numel()一致，相乘，prod(a.size) == prod(b.size)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
当然，我们的变换也要符合物理意义，不然随意的变换会污染数据
例如，1*28*28的1通道28*28像素的图片可变换为由784个单维度的向量表示
'''
print('-------------------View reshape-----------------------')
a = torch.rand(4, 1, 28, 28)
print(a.shape)

print(a.view(4, 28 * 28))
print(a.view(4, 28 * 28).shape)
print(torch.reshape(a, (4, 28 * 28)).shape)

print(a.view(4 * 28, 28).shape)  # 只关注行

print(a.view(4 * 1, 28, 28).shape)  # 合并通道，只关注每张图片

b = a.view(4, 784)  # 容易丢失维度信息，所以要时刻记住
print(b.view(4, 28, 28, 1).shape)
print(b.view(4, 1, 28, 28).shape)

'''
Squeeze v.s. unsqueeze
维度增加和维度减小，增加一个dim=1的维度，减去dim=1的维度
'''
'''
unsqueeze

unsqueeze扩展维度原则：
正数索引之前，负数索引之后
正数索引： 0   1  2   3
a.shape: 4   1  28  28
负数索引：-4  -3 -2  -1

a.unsqueeze(dim) <==> torch.unsqueeze(a, dim)
'''
print('-------------------unsqueeze-----------------------')
# a,[-5, 4]
print(torch.unsqueeze(a, 0).shape)  # 在0维之前增加一个维度
print(torch.unsqueeze(a, -1).shape)  # 在-1维之后增加一个维度

print(torch.unsqueeze(a, 4).shape)  # 4之前，及3之后
print(torch.unsqueeze(a, -4).shape)  # -4之后，及1之前
print(torch.unsqueeze(a, -5).shape)  # -5之后

print('-------------------For example-----------------------')
'''
假设目前要为shape为[4, 32, 14, 14]的图片的通道加上偏置[32]
我们则需要把偏置的维度变换成图片的维度
'''
b = torch.rand(32)
f = torch.rand(4, 32, 14, 14)
# f + b
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape)

'''
Squeeze
squeeze默认会把dim为1的部分剪掉，也可以指定dim，若为1则剪掉，否则保持不变
'''
print('-------------------Squeeze-----------------------')
print(b.shape)
print(b.squeeze().shape)
print(b.squeeze(0).shape)
print(b.squeeze(-1).shape)
print(b.squeeze(-4).shape)

'''
Expand / repeat
维度扩展 / 拷贝维度
不主动数据 / 增加数据
推荐 / 不推荐（Memory touched）
只能 1->N，不能 M->N
若为-1则保持当前维度不变
改变的是shape
'''
print('-------------------Expand-----------------------')
a = torch.rand(4, 32, 14, 14)
print(b.shape)
print(b.expand(4, 32, 14, 14).shape)
print(b.expand(-1, 32, -1, -1).shape)
print(b.expand(-1, 32, -1, -4).shape)  # 这是个bug，不建议使用

print('-------------------repeat-----------------------')
print(b.shape)
print(b.repeat(4, 32, 1, 1).shape)
print(b.repeat(4, 1, 1, 1).shape)
print(b.repeat(4, 1, 32, 32).shape)

'''
转置
'''
'''
t()只适用于dim = 2的
'''
print('-------------------t()-----------------------')
a = torch.randn(3, 4)
print(a)
print(a.t())

'''
Transpose
用的时候要注意跟踪维度
交换维度后数据会一般变得不连续，这时，需要用contiguous使数据连续
'''
print('-------------------Transpose-----------------------')
a = torch.rand(4, 3, 32, 32)
# a1 = a.transpose(1, 3).view(4, 3*32*32).view(4*3*32*32)

# [b, c, H, W] -> [b, W, H, c] -> [b, c, W, H]  数据维度顺序改变了
a1 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 3, 32, 32)

# [b, c, H, W] -> [b, W, H, c] --后三维度压缩--> [b, (W, H, c)] --按照压缩前的结构还原--> [b, W, H, c] --W和c换--> [b, c, H, W]  数据维度与初始值顺序一致
a2 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 32, 32, 3).transpose(3, 1)

print(torch.all(torch.eq(a, a1)))
print(torch.all(torch.eq(a, a2)))

'''
permute
对维度任意重拍，参数对应的索引项为原来数据绑定项的重排
例如：
b = torch.rand(4, 3, 28, 32)
b.permute(0, 2, 3, 1)  -->  torch.Size([4, 28, 32, 3])
表示原来的dim0上的4放在0位置，原来的dim2上的28放在1位置，原来的dim3上的32放在2位置，原来的dim1上的3放在3位置
'''
print('-------------------permute-----------------------')
b = torch.rand(4, 3, 28, 32)

print(b.transpose(1, 3).shape)
print(b.transpose(1, 3).transpose(1, 2).shape)

print(b.permute(0, 2, 3, 1).shape)
