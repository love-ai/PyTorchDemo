import torch


def test_cat():
    A = torch.ones(3, 3)
    B = 2 * torch.ones(3, 3)
    # C = torch.cat((A, B), 1)
    C = torch.stack((A, B), 0)
    D = torch.stack((A, B), 1)
    E = torch.stack((A, B), 2)
    print(A)
    print(B)
    print(C)
    print(D)
    print(E)


def test_stack():
    A = torch.arange(0, 4)
    B = torch.arange(5, 9)
    C = torch.stack((A, B), 0)
    D = torch.stack((A, B), 1)
    print(A)
    print(B)
    print(C)
    print(D)


# 切分成确定的份数
def test_chunk():
    A = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    B = torch.chunk(A, 2, 0)
    C = torch.chunk(A, 3, 0)
    # 超过的话就全部切开
    F = torch.chunk(A, 20, 0)
    print(A)
    print(B)
    print(C)
    print(F)
    D = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    E = torch.chunk(D, 4, 0)
    print(D)
    print(E)


# 按照每份确定的数量进行拆解
def test_split():
    A = torch.rand(4, 4)
    print(A)
    B = torch.split(A, 2, 0)
    print(B)
    C = torch.rand(5, 4)
    print(C)
    # 当split_size_or_sections 为列表时 按照列表每项的个数进行拆分
    D = torch.split(C, (3, 1, 1), 0)
    print(D)


# 将torch 按照维度降维拆解 unbind 是一种降维切分的方式，相当于删除一个维度之后的结果。
def test_unbind():
    A = torch.arange(0, 27).view(3, 3, 3)
    print(A)
    B = torch.unbind(A, 0)
    print(B)


# 是基于给定的索引来进行数据提取的
def test_index_select():
    A = torch.arange(0, 16).view(4, 4)
    print(A)
    # index是 torch.Tensor 类型
    B = torch.index_select(A, 0, torch.tensor([2, 3]))
    print(B)
    C = torch.index_select(A, 1, torch.tensor([1, 2]))
    print(C)


# 根据条件去取对应的数据
def test_masked_select():
    A = torch.rand(5)
    print(A)
    B = torch.masked_select(A, A > 0.3)
    print(B)


def after_class_test():
    A = torch.tensor([[4, 5, 7], [3, 9, 8], [2, 3, 4]])

    B = torch.tensor([[True, False, False], [True, True, False], [False, False, True]])
    C = torch.masked_select(A, B)
    print(C)

    D = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    E = torch.masked_select(A, D > 0)
    print(E)


if __name__ == '__main__':
    # test_cat()
    # test_stack()

    # test_chunk()
    # test_split()
    # test_unbind()
    # test_index_select()
    test_masked_select()

    # after_class_test()
