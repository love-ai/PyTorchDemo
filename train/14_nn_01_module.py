import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn

# 生成测试数据
from visdom import Visdom


def test1():
    w = 2
    b = 3
    xlim = [-10, 10]
    x_train = np.random.randint(low=xlim[0], high=xlim[1], size=100)
    # print(x_train)
    y_train = [w * x + b + random.randint(0, 2) for x in x_train]
    # print(y_train)
    plt.plot(x_train, y_train, 'bo')
    plt.show()
    return {x_train, y_train}


# 模型构建
class LinearModel(nn.Module):

    # 初始化方法
    def __init__(self):
        super().__init__()  # 必须有
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    # 向前传播
    def forward(self, input):
        return (input * self.weight) + self.bias


def test():
    # 生成测试数据
    w = 2
    b = 3
    xlim = [-10, 10]
    x_train = np.random.randint(low=xlim[0], high=xlim[1], size=100)
    y_train = [w * x + b + random.randint(0, 1) for x in x_train]
    # 创建模型
    model = LinearModel()
    # 创建优化器 设置优化参数
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
    # 转换为tensor
    y_train = torch.tensor(y_train, dtype=torch.float32)
    # 转换x_train为tensor
    input = torch.from_numpy(x_train)
    print("初始模型：" + str(model.state_dict()) + "\n\n")

    writer = SummaryWriter()
    # 实例化一个窗口
    viz = Visdom(port=8097)
    # 初始化窗口的信息
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

    for n_iter in range(1000):
        # 计算模型的全量输出
        output = model(input)
        # 将当前参数计算的全量输出和实际值带入 计算MSE损失函数
        loss = nn.MSELoss()(output, y_train)
        # 要通过zero_grad()函数把梯度清零，不然PyTorch每次计算梯度会累加，不清零的话第二次算的梯度等于第一次加第二次的
        model.zero_grad()
        # 反向梯度传播 这个过程之后梯度会记录在变量中
        loss.backward()
        # 用计算的梯度去做优化
        optimizer.step()
        writer.add_scalar("Loss/train", loss, n_iter)
        viz.line([loss.item()], [n_iter], win='train_loss', update='append')
        print('Loss {}'.format(loss))
        print("训练结果：" + str(model.state_dict()) + "\n\n")

    for parameter in model.named_parameters():
        print(parameter)

    # print(model.state_dict())
    # 保存模型 只保存参数
    # torch.save(model.state_dict(), "./model/linear_model_dict.pth")
    # 保存模型 保存网络结构和训练后的参数
    # torch.save(model, "./model/linear_model_all.pth")


# 通过保存模型的参数进行加载
def load_model_by_dict():
    # 先创建网络结构
    linear_model = LinearModel()
    # 加载保存的参数
    linear_model.load_state_dict(torch.load("./model/linear_model_dict.pth"))
    # 让模型进入评估
    linear_model.eval()
    for parameter in linear_model.named_parameters():
        print(parameter)
    # 使用真实数据来测试模型输出
    print(linear_model(30))


# 加载整个模型
def load_model_by_all():
    model = torch.load("./model/linear_model_all.pth")
    # 同样告诉模型要开始评估了
    model.eval()
    for parameter in model.named_parameters():
        print(parameter)
    print(model(10))


if __name__ == '__main__':
    # test1()
    test()
    # load_model_by_dict()
    # load_model_by_all()
