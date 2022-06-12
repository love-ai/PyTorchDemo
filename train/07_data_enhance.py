import torchvision.transforms
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def data_transform():
    img = Image.open("img/jike.jpg")
    # img.show()
    print(type(img))

    # 将PIL.Image 转换成Tensor
    img1 = transforms.ToTensor()(img)
    print(type(img1))

    # 将Tensor转成 PIL.Image
    img2 = transforms.ToPILImage()(img1)
    img2.show()
    print(type(img2))


def transform_resize():
    resize_img_oper = transforms.Resize((200, 200), InterpolationMode.BILINEAR)
    origin_img = Image.open("img/jike.jpg")
    origin_img.show()
    img = resize_img_oper(origin_img)
    img.show()


# 剪裁 旋转
def transform_crop_flip():
    # 定义操作
    center_crop_oper = transforms.CenterCrop((100, 100))
    random_crop_oper = transforms.RandomCrop((50, 50))
    five_crop_oper = transforms.FiveCrop((50, 150))
    h_flip_oper = transforms.RandomHorizontalFlip(p=1)
    v_flip_oper = transforms.RandomVerticalFlip(p=1)

    origin_img = Image.open("img/jike.jpg")
    img1 = center_crop_oper(origin_img)
    img2 = random_crop_oper(origin_img)
    imgList = five_crop_oper(origin_img)
    # img1.show()
    # img2.show()
    # for img in imgList:
    #     img.show()

    # img3 = h_flip_oper(origin_img)
    # img4 = v_flip_oper(origin_img)
    # img3.show()
    # img4.show()

    m_transform = transforms.Compose([center_crop_oper, h_flip_oper])
    transform_img = m_transform(origin_img)
    print(transform_img)
    transform_img.show()


def make_normalize():
    origin_img = Image.open("img/jike.jpg")
    origin_img.show()
    norm_oper = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # 图像转化为tensor
    img_tensor = transforms.ToTensor()(origin_img)
    # 标准化
    norm_tensor = norm_oper(img_tensor)
    # tensor转换成pil
    pil_img = transforms.ToPILImage()(norm_tensor)
    pil_img.show()


def use_with_datasets():
    # 定义comose
    my_trans = transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomHorizontalFlip(p=1),
                                   transforms.Normalize((0.5), (0.5))
                                   ])
    mnist_dataset = torchvision.datasets.MNIST(root="./data",
                                               train=False,
                                               transform=my_trans,
                                               target_transform=None,
                                               download=True)
    item = mnist_dataset.__getitem__(0)
    print(type(item[1]))
    img = transforms.ToPILImage()(item[0])
    img.show()


if __name__ == '__main__':
    # data_transform()
    # transform_resize()
    # transform_crop_flip()
    make_normalize()
    # use_with_datasets()
