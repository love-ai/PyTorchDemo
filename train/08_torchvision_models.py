import torch.nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import models, transforms


def test_googlenet():
    # 随机初始化的权重，创建一个 GoogLeNet 模型 需要经过训练才能使用
    # random_googlenet = models.googlenet()
    # 直接导入训练好的模型来使用
    # Downloading: "https://download.pytorch.org/models/googlenet-1378be20.pth" to /Users/xiaowei/.cache/torch/hub/checkpoints/googlenet-1378be20.pth
    # 100.0%
    pretrained_googlenet = models.googlenet(pretrained=True)

    # vgg16 = models.vgg16(pretrained=True)

    print("finish")

    # 模型微调
    # 分类层的输入参数
    in_features = pretrained_googlenet.fc.in_features
    print("in_features:", in_features)
    # 分类层的输出参数
    out_features = pretrained_googlenet.fc.out_features
    print("out_features:", out_features)

    # 修改分类层的输出参数为 10
    pretrained_googlenet.fc = torch.nn.Linear(in_features, 10)
    # 更新后的分类层的输出参数
    new_out_features = pretrained_googlenet.fc.out_features
    print("new_out_features:", new_out_features)


def test_make_grid():
    my_dataset = torchvision.datasets.MNIST(root="./data",
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            target_transform=None,
                                            download=True)

    tensor_dataloader = DataLoader(dataset=my_dataset, batch_size=32)
    # for img_tensor, label_tensor in tensor_dataloader:
    #     print(img_tensor, label_tensor)
    data_iter = iter(tensor_dataloader)
    img_tensor, label_tensor = data_iter.next()
    print(img_tensor.shape)

    grid_tensor = torchvision.utils.make_grid(img_tensor, nrow=8, padding=28)
    img = transforms.ToPILImage()(grid_tensor)
    img.show()
    # torchvision.utils.save_image(grid_tensor, 'img/grid1.jpg')
    # 输入为List 调用grid_img函数后保存
    torchvision.utils.save_image(img_tensor, 'img/grid2.jpg', nrow=5, padding=2)


if __name__ == '__main__':
    test_googlenet()
    # test_make_grid()
