import torch
from torch import nn


# 深度可分离卷积（Depthwise Separable Convolution）用来处理轻量化模型
# 效率为1/k^2 k为卷积核的边长
# dw pw 卷积 以下例子已跑通 可以运行多看看
def test_dw_pw():
    # 生成输入通道数据 unsqueeze(0) 是因为第一个维度为batch_size
    x = torch.tensor([[[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5]],
                      [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5]],
                      [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5]]],
                     dtype=torch.int16).unsqueeze(0)
    # x = torch.rand((3, 5, 5)).unsqueeze(0)
    print(x.shape)
    print(x)
    # 输入通道为3个图层 3个通道 因为第一个维度是batch_size 所以这里的通道数取[1]
    in_channels_dw = x.shape[1]
    out_channels_dw = x.shape[1]
    # 卷积核大小
    kernel_size = (3, 3)
    # 步长
    stride = 1
    # 生成dw卷积 分组=输入渠道的通道数，这样就是每个通道都分别和对饮过的卷积核的单独通道进行计算，输出通道数=输入通道数
    dw = nn.Conv2d(in_channels_dw, out_channels_dw, kernel_size, stride, groups=in_channels_dw)
    # 指定dw卷积核
    kernels = torch.tensor(
        [[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
         [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
         [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]],
        dtype=torch.int16)
    print(kernels.shape)
    print(kernels)
    dw.weight = nn.Parameter(kernels, requires_grad=False)
    print(dw)
    with torch.no_grad():
        print(dw(x))
    in_channels_pw = out_channels_dw
    out_channels_pw = 4
    kernel_size_pw = 1
    # pw卷积就是标准卷积 不用设置group 默认为1
    pw = nn.Conv2d(in_channels_pw, out_channels_pw, kernel_size_pw, stride)
    # 指定pw卷积核 通道数为输出通道数，dw计算的结果的每一个通道都会与pw中的一个通道做计算并生成一个新的通道
    kernel_pw = torch.tensor([[[[1]], [[1]], [[1]]],
                              [[[2]], [[2]], [[2]]],
                              [[[3]], [[3]], [[3]]],
                              [[[4]], [[4]], [[4]]]], dtype=torch.int16)
    pw.weight = nn.Parameter(kernel_pw, requires_grad=False)
    print(pw)
    print(pw.weight)
    with torch.no_grad():
        out = pw(dw(x))
    print(out.shape)
    print(out)


if __name__ == '__main__':
    test_dw_pw()
