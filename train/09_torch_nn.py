import torch
from torch import nn


def test_conv2d():
    # Pytorch 输入 tensor 的维度信息是 (batch_size, 通道数，高，宽) 所以这里需要是四个维度
    input_feat = torch.tensor([[4, 1, 7, 5], [4, 4, 2, 5], [7, 7, 2, 4], [1, 0, 2, 4]], dtype=torch.int8).unsqueeze(
        0).unsqueeze(0)
    print(input_feat)
    print(input_feat.shape)
    # 输入通道1 输出通道1 卷积核大小（2，2） 滑动步长1 输出原图一样大小，使用偏移项目
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=1, padding="same", bias=False)
    # 不指定卷积核则随机
    kernels = torch.tensor([[[[1, 0], [2, 1]]]], dtype=torch.int8)
    print(kernels.shape)
    conv2d.weight = nn.Parameter(kernels, requires_grad=False)

    # 卷积核
    print(conv2d.weight)
    # 偏移项
    print(conv2d.bias)

    # 输出
    output = conv2d(input_feat)
    print(output)


if __name__ == '__main__':
    test_conv2d()
