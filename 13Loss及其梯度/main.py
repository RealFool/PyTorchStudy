import torch
from torch.nn import functional as F

'''
Typical Loss
    ▪ Mean Squared Error
    ▪ Cross Entropy Loss
        ▪ binary
        ▪ multi-class
        ▪ +softmax
        ▪ Leave it to Logistic Regression Part
'''
'''
autograd.grad
两种方法
torch.autograd.grad(loss, [w1, w2,…])
▪ [w1 grad, w2 grad…]
loss.backward()
▪ w1.grad
▪ w2.grad
'''
print('-------------------mse_loss,求导1-----------------------')
x = torch.ones(1)
# 不加requires_grad=True会报RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
w = torch.full([1], 2.0, requires_grad=True)
print(x, w)
# d动态图
# mse = (-1*w)^2, w=2
# (input, output), pre,label
mse = F.mse_loss(x*w, torch.ones(1))
print(mse)
print(torch.autograd.grad(mse, [w]))

print('-------------------mse_loss,求导2-----------------------')
x = torch.ones(1)
# 不加requires_grad=True会报RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
w = torch.full([1], 2.0, requires_grad=True)
print(x, w)
mse = F.mse_loss(x*w, torch.ones(1))
# 方向传播，会把梯度信息返回给每个成员变量的grad中
mse.backward()
print(w.grad)

