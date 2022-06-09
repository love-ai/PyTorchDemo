import torch


def testTensor():
    # a = torch.zeros(2, 3, 5)
    # print(a)
    # print(a.shape)
    # print(a.size())
    # print(a.numel())

    b = torch.rand(2, 3, 5) * 100
    print(b.shape)
    print(b)
    # 打印 1，0，0的值和内存地址
    print(b[1, 0, 0])
    # 打印地址 打印的是整个tensor的地址而不是某一个数据项的地址
    print(id(b[1, 0, 0]))
    # permute 可以交换多个维度的数据，第一个参数代表的是之前数组第几个维度的数据
    b = b.permute(2, 1, 0)
    print(b.shape)
    print(b)
    print(b[0, 0, 1])
    # 打印地址 发现内存地址确实变的和上面不一样了
    print(id(b[0, 0, 1]))

    # transpose 一次只可以交换两个维度的数据
    # b = b.transpose(1, 0)
    # print(b.shape)
    # print(b)
    # 不管是permute还是transpose变换之后的位置的数据就不连续了


def reshape():
    a = torch.randn(4, 4)
    print(a)
    print(a.shape)
    a = a.view(2, 8)
    print(a)
    print(a.shape)
    # 修改地址使其不连续
    a = a.permute(1, 0)
    print(a)
    print(a.shape)
    # RuntimeError: view size is not compatible with input tensor's size and stride
    # (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
    # a = a.view(4, 4)
    # reshape 相当于进行了两步操作，先把 Tensor 在内存中捋顺了，然后再进行 view 操作。
    a = a.reshape(4, 4)
    # 这个输出和原始的就不一样了
    print(a)


if __name__ == '__main__':
    # testTensor()
    reshape()
