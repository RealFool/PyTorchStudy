import torch

'''
索引与切片
'''
print('-------------------Indexing-----------------------')
a = torch.rand(4, 3, 28, 28)  # 想象为一组彩色图片
print(a[0].shape)   # 第一张图片

print(a[0, 0].shape)    # 第一张图片的第一个通道

print(a[0, 0, 2, 4])    # 第一张图片的第一个通道上的第2行第4列的像素，标量

'''
select first/last N
切片
当切片索引为正值时
: 可看作 -> ，及 start : end 等价为 start -> end 并且满足[start, end)
当切片索引为负值时
: 可看作 <- ，及 end : start 等价为 end <- start 并且满足[start, end)
'''
print('-------------------select first/last N-----------------------')
print(a.shape)
# : 可看作 -> ，及 start : end 等价为 start -> end 并且满足[start, end)
print(a[:2].shape)

print(a[:2, :1, :, :].shape)

print(a[:2, 1:, :, :].shape)

print(a[:2, -1:, :, :].shape)
