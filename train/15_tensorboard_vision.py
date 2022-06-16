import numpy as np
from PIL import Image
from numpy import shape
from torch.utils.tensorboard import SummaryWriter


# 生成测试数据后， 去代码的当前目录执行：
# tensorboard --logdir=runs 然后浏览器中查看
# tensorboard 只能在训练完成后查看训练的曲线

def test_tensorboard():
    writer = SummaryWriter()
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    img = np.zeros((3, 100, 100))
    img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    # img1 = img.reshape((100, 100, 3)) # 只有reshap成H W 通道数 才能转成image
    # pic = Image.fromarray(np.uint8(img1))
    # pic.show()
    writer.add_image('my_image', img, 0)
    writer.close()


if __name__ == '__main__':
    test_tensorboard()
