import torch
from matplotlib import pyplot as plt


# loss 下降曲线绘制
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


# 画图，图像识别，可视化识别结果
def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        # 子图
        plt.subplot(2, 3, i + 1)
        # tight_layout会自动调整子图参数，使之填充整个图像区域
        plt.tight_layout()
        # plt.imshow()作用就是展示一副热度图，将数组表示为一幅图，interpolation:此参数是用于显示图像的插值方法。
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


'''
torch.Tensor.scatter_()是torch.gather()函数的方向反向操作。两个函数可以看成一对兄弟函数。gather用来解码one hot，scatter_用来编码one hot。

'''
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out
