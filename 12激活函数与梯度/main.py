import torch

"""
Sigmoid
sigmoid(x) = 1/(1+e**(-1))
Derivative
sigmoid(1 - sigmoid)
"""
print('-------------------------Sigmoid---------------------------')
a = torch.linspace(-100, 100, 10)
print(a)

print(torch.sigmoid(a))

'''
Tanh
tanh(x)=2sigmoid(x) − 1
Derivative
1 - (tanh(x))**2
'''
print('-------------------------Tanh---------------------------')
a = torch.linspace(-1, 1, 10)
print(a)
print(torch.tanh(a))
'''
Rectified Linear Unit
ReLU
'''
print('-------------------------ReLU---------------------------')
from torch.nn import functional as F

a = torch.linspace(-1, 1, 10)
print(torch.relu(a))

print(F.relu(a))
