import time

import numpy as np
import matplotlib.pyplot as plt


def test():
    # 一维数组
    arr_1_d = np.asarray(["1", "2"])
    # 二维数组
    arr_2_d = np.asarray([["1", "2"], ["3", "4"]])
    arr_3_d = np.asarray([[["1", "2", "2.5"], ["3", "4", "4.5"], ["3", "4", "5.5"]],
                          [["1", "2", "3.5"], ["3", "4", "6.5"], ["3", "4", "7.5"]],
                          [["1", "2", "8.5"], ["3", "4", "9.5"], ["3", "4", "1.5"]]])

    # print(arr_1_d)
    # print(arr_2_d)
    # print(arr_3_d)
    # print(arr_1_d.ndim)
    # print(arr_2_d.ndim)
    # print(arr_3_d.ndim)
    print(arr_1_d.shape)
    print(arr_2_d.shape)
    print(arr_3_d.shape)

    # reshape 不会修改原数组
    print(arr_2_d.reshape(4, 1))
    # 修改成长方体
    print(arr_3_d.reshape(3, 9, 1))
    # 修改成二维数组
    print(arr_3_d.reshape(3, 9, 1))

    # 从0生成20哥数据，然后reshap成4行5列，默认按照行优先"C"进行遍历
    arr20 = np.arange(20).reshape(4, 5)
    print(arr20)
    # 打印size
    print(arr20.size)
    # 输出数组类型
    print(arr20.dtype)
    print(arr_1_d.dtype)
    # 可以直接对数组乘除
    print(np.ones(shape=(2, 3), dtype="int32") * 5)
    print(np.zeros(shape=(9, 9), dtype="int32"))
    # 设置间隔默认1
    print(np.arange(3, 300, 23))
    # 设置个数
    print(np.linspace(start=3, stop=300, num=10, dtype="int32", endpoint=True, retstep=True))


def test_py_plot():
    X = np.arange(-50, 51, 1)
    # X = np.linspace(start=-50, stop=51, num=15, dtype="int32")
    Y = X ** 2
    plt.plot(X, Y, color="blue")
    plt.legend()
    plt.show()


def test_axis():
    # arr = np.asarray([[5, 7, 9], [8, 5, 0], [6, 4, 3], [1, 7, 4]], dtype="int64")

    arr1 = np.arange(1600000000).reshape(40000, 40000)
    # print(arr1)
    start = time.time()
    print(np.average(arr1, 1))
    # 11.237936735153198
    print("1----" + str(time.time() - start))
    print(arr1.sum(1) / 3)
    # 24.719791650772095
    print("2----" + str(time.time() - start))


if __name__ == '__main__':
    # test()
    # test_py_plot()
    test_axis()
