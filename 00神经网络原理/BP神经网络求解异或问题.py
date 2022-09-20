# -*- coding: utf-8 -*-
import numpy as np


# 双曲正切函数,该函数为奇函数
def tanh(x):
    return np.tanh(x)


# tanh导函数性质:f'(t) = 1 - f(x)^2
def tanh_prime(x):
    return 1.0 - tanh(x) ** 2


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :参数layers: 神经网络的结构(输入层-隐含层-输出层包含的结点数列表)
        :参数activation: 激活函数类型
        """
        if activation == 'tanh':  # 也可以用其它的激活函数
            self.activation = tanh
            self.activation_prime = tanh_prime
        else:
            pass

        # 存储权值矩阵
        self.weights = []

        # range of weight values (-1,1)
        # 初始化输入层和隐含层之间的权值
        print('------------------', len(layers))    # 3
        for i in range(1, len(layers) - 1):
            # layer[i-1]+1 = layer[0]+1 = 2+1 = 3
            # layers[i] + 1 = layer[1]+1 = 3
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1  # add 1 for bias node
            print('---------初始化输入层和隐含层之间的权值---------')
            print(r)
            self.weights.append(r)

        # 初始化输出层权值
        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        print('---------初始化输出层权值---------')
        print(r)
        self.weights.append(r)
        print('---------所有权重---------')
        print(self.weights)

    def fit(self, X, Y, learning_rate=0.2, epochs=10000):
        # 将一列一加到X
        # 这是为了将偏置单元添加到输入层
        # np.hstack()将两个数组按水平方向组合起来, 4*2 --> 4*3
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        print('----将一列一加到X,将偏置单元添加到输入层----')
        print(X)

        for k in range(epochs):  # 训练固定次数
            # if k % 1000 == 0: print('epochs:', k)

            # 从区间中的离散均匀分布返回随机整数 [0, low).[0, 4)
            i = np.random.randint(X.shape[0], high=None)
            a = [X[i]]  # 从m个输入样本中随机选一组
            # len(self.weights) 2
            for l in range(len(self.weights)):
                # 每组输入样本(第一列未偏执值b)与权值进行矩阵相乘,a:1*3, weights:3*3 ---> 1*3
                dot_value = np.dot(a[l], self.weights[l])  # 权值矩阵中每一列代表该层中的一个结点与上一层所有结点之间的权值
                activation = self.activation(dot_value)     # 放入激活函数
                a.append(activation)
                print(l)

            # 反向递推计算delta:从输出层开始,先算出该层的delta,再向前计算
            error = Y[i] - a[-1]  # 计算输出层delta
            deltas = [error * self.activation_prime(a[-1])]

            # 从倒数第2层开始反向计算delta
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))

            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()  # 逆转列表中的元素

            # backpropagation
            # 1. Multiply its output delta and input activation to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):  # 逐层调整权值
                layer = np.atleast_2d(a[i])  # View inputs as arrays with at least two dimensions
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * np.dot(layer.T, delta)  # 每输入一次样本,就更新一次权值

    def predict(self, x):
        a = np.concatenate((np.ones(1), np.array(x)))  # a为输入向量(行向量)
        for l in range(0, len(self.weights)):  # 逐层计算输出
            a = self.activation(np.dot(a, self.weights[l]))
        return a


if __name__ == '__main__':
    nn = NeuralNetwork([2, 2, 1])  # 网络结构: 2输入1输出,1个隐含层(包含2个结点)

    X = np.array([[0, 0],  # 输入矩阵(每行代表一个样本,每列代表一个特征)
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([0, 1, 1, 0])  # 期望输出

    nn.fit(X, Y)  # 训练网络

    print('w:', nn.weights)  # 调整后的权值列表

    for s in X:
        print(s, nn.predict(s))  # 测试
