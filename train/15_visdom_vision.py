import time

import numpy as np
from visdom import Visdom


# visdom 可以实时显示训练数据
# 执行 python -m visdom.server
def test_visdom():
    # 将窗口实例化
    viz = Visdom()
    viz.line([[0., 0.]], [0], win='train', opts=dict(title='train_loss', legend=["loss", "acc"]))
    for n_iter in range(10):
        # 随机获取loss
        loss = 0.2 * np.random.randn() + 1
        accuracy = 0.1 * np.random.randn() + 0.5
        # 更新图像
        viz.line([[loss, accuracy]], [n_iter], win='train', update= 'append')
        time.sleep(1)
    # img = np.zeros((3, 100, 100))
    # img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    # img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    # # 可视化图像
    # viz.image(img)


if __name__ == '__main__':
    test_visdom()
