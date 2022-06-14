from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from numpy import shape


def test1():
    img = Image.open("jike.jpg")
    img_arr = np.asarray(img)
    print(shape(img_arr))
    # 索引与切片
    arr0 = img_arr[:, :, 0]
    arr1 = img_arr[:, :, 1]
    arr2 = img_arr[:, :, 2]
    print(arr0.shape)
    white_arr = np.zeros((img_arr.shape[0], img_arr.shape[1], 2))
    print(white_arr.shape)

    red_arr_new_axis = arr0[:, :, np.newaxis]
    print(arr0[0][0])
    # 已经将数据放到第三个维度中
    print(red_arr_new_axis[0][0][0])

    # 叠加两个数组 需要注意叠加顺序及叠加的轴的方向
    red_arr = np.concatenate((red_arr_new_axis, white_arr), 2)

    # 通过直接赋值的方式进行合并
    blue_arr = np.zeros(img_arr.shape)
    green_arr = np.zeros(img_arr.shape)
    green_arr[:, :, 1] = arr1
    blue_arr[:, :, 2] = arr2

    # 通过pyplot绘制到一起
    # plt.subplot(2, 2, 1)
    # plt.title('Origin Image')
    # plt.imshow(img_arr)
    # plt.axis('off')
    # plt.subplot(2, 2, 2)
    # plt.title('Red Channel')
    # plt.imshow(red_arr.astype(np.uint8))
    # plt.axis('off')
    # plt.subplot(2, 2, 3)
    # plt.title('Green Channel')
    # plt.imshow(green_arr.astype(np.uint8))
    # plt.axis('off')
    # plt.subplot(2, 2, 4)
    # plt.title('Blue Channel')
    # plt.imshow(blue_arr.astype(np.uint8))
    # plt.axis('off')
    # plt.savefig('./rgb_pillow.png', dpi=150)
    print("---------")
    print(shape(red_arr))
    # 转换回Image然后输入
    im = Image.fromarray(np.uint8(red_arr))
    im.show()
    im.save("red.jpg")


def test_copy():
    img = Image.open("jike.jpg")
    print(img.mode)
    img_arr = np.asarray(img)
    img_arr[:, :, 1:] = 0
    # print(img_arr)
    print(img_arr.flags)

    # a = np.arange(20)
    # print(a.shape)
    # print(a)
    # b = a.view()
    # print(b.shape)
    # print(b)
    # b.shape = 4, 5
    # print(b)
    # b[0, 0] = 123
    # print(b)
    # print(a)


def testMaxAndSort():
    probs = np.array([0.075, 0.15, 0.075, 0.15, 0.0, 0.05, 0.05, 0.2, 0.25])
    # 求最大的数的下标
    print(np.argmax(probs))
    # 求最小数的小标
    print(np.argmin(probs))
    probs_idx_sort = np.argsort(-probs)
    # 按照从小到大排序 返回下标
    print(probs_idx_sort)
    # 返回最前面的三个probs_idx_sort[0:3] 0可以省略
    print(probs_idx_sort[:3])


def test_mask():
    scores = np.random.rand(256, 256, 2)
    scores[:, :, 1] = 1 - scores[:, :, 0]
    print(scores)
    # argmax 因为返回的是下标 正好是0 和 1
    mask = np.argmax(scores, axis=2)
    # print(mask)
    mask1 = (scores[:, :, 0] < scores[:, :, 1]).astype('int')
    print(mask1)


if __name__ == '__main__':
    test1()
    # test_copy()
    # testMaxAndSort()

    # test_mask()
