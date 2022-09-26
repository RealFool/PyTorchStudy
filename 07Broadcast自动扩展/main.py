import torch

'''
在dim为1的维度进行自动扩张
unsqueeze + Expand

规定右边是小维度，左边是大维度
小维度指定（维度与上边的维度相同），大维度随意

▪ Match from Last dim!
    ▪ If current dim=1, expand to same
    ▪ If either has no dim, insert one dim and expand to same
    ▪ otherwise, NOT broadcasting-able
'''