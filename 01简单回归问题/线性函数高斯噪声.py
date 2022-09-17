import random

import numpy as np
from matplotlib import pyplot as plt


def gauss_noisy(x, y):
    """
    对输入数据加入高斯噪声
    :param x: x轴数据
    :param y: y轴数据
    :return:
    """
    mu = 0
    # 高斯噪声sigma越大，离散越大
    sigma = 2
    for i in range(len(x)):
        x[i] += random.gauss(mu, sigma)
        y[i] += random.gauss(mu, sigma)


if __name__ == '__main__':
    # 在0-50的区间上生成500个点作为测试数据
    xl = np.linspace(0, 50, 500, endpoint=True)
    yl = 2 * xl + 5

    # 加入高斯噪声
    gauss_noisy(xl, yl)

    # 初始化一个空Dataframe
    import pandas as pd

    data_frame = pd.DataFrame(
        columns=['x', 'y'], index=[])

    # 插入一行，如果需要插入多行，加个for循环即可
    for i in range(len(xl)):
        singlelist = [xl[i], yl[i]]
        indexsize = data_frame.index.size
        data_frame.loc[indexsize] = singlelist
        data_frame.index = data_frame.index + 1

    data_frame.to_csv('data_frame.csv', index=False)

    # 画出这些点
    plt.plot(xl, yl, linestyle='', marker='.')
    plt.plot(xl, 2.126028567945425 * xl + 0.3392305175620397, 'r-', linewidth=2)
    plt.show()

