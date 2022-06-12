import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image


# 通过一个图片测试alexnet模型
def test_alexnet_model():
    alexnet = models.alexnet()
    # 打印 网络结构 修改 全链接层的参数
    alexnet.load_state_dict(torch.load("./model/alexnet-owt-7be5be79.pth"))
    print(alexnet)

    img = Image.open('img/dog.jpg')
    transform1 = transforms.Compose([transforms.Resize((224, 224))])  # 这里建议Resize
    transform = transforms.Compose(
        [transforms.RandomResizedCrop((224, 224)),  # RandomResizedCrop 和CenterCrop确实是按照像素1：1的去剪裁的 图会变得很小
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform1(img).show()
    input_tensor = transform(img).unsqueeze(0)
    # 得到模型输出结果
    res = alexnet(input_tensor)
    # 打印类型
    print(res.argmax())  # tensor(263) 柯基狗


# 加载cifar10数据 并打印出来
def load_data():
    cifar10_dataset = torchvision.datasets.CIFAR10(root="./data",
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   target_transform=None,
                                                   download=True)

    tensor_dataloader = DataLoader(dataset=cifar10_dataset, batch_size=256)
    data_iter = iter(tensor_dataloader)
    img_tensor, label_tensor = data_iter.__next__()
    print(img_tensor.shape)
    grid_tensor = torchvision.utils.make_grid(img_tensor, nrow=16, padding=2)
    img = transforms.ToPILImage()(grid_tensor)
    img.show()
    img.save("./img/cifar10_train_gird.jpg")


# 修改模型的默认输出分类数
def get_modify_classifier():
    alexnet = models.alexnet()
    alexnet.load_state_dict(torch.load("./model/alexnet-owt-7be5be79.pth"))

    for parameter in alexnet.parameters():
        parameter.requires_grad = False

    # 打印 网络结构 修改 全链接层的参数
    # print(alexnet)

    # 开始修改全链接层 输出的分类数 对于CIFAR10 我们只需要10个分类
    # 提取分类层的输入参数
    fc_in_features = alexnet.classifier[6].in_features
    alexnet.classifier[6] = torch.nn.Linear(fc_in_features, 10)
    # 再次打印看看修改效果
    # print(alexnet)
    return alexnet


def start_train():
    # 获取修改模型
    alexnet = get_modify_classifier()

    # 均值[0.485, 0.456, 0.406]，标准差[0.229, 0.224, 0.225]是ImageNet的均值与标准差。torchvision中的模型都是在ImageNet上训练的
    transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # 初始化数据集
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                   train=False,
                                                   transform=transform,
                                                   target_transform=None,
                                                   download=True)
    # 设置数据加载
    dataloader = DataLoader(
        dataset=cifar10_dataset,  # 传入的数据集, 必须参数
        batch_size=32,  # 输出的batch大小
        shuffle=True,  # 数据是否打乱˙
        num_workers=2)  # 进程数, 0表示只有主进程

    # 定义优化器
    optimizer = torch.optim.SGD(alexnet.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

    for epoch in range(3):
        for item in dataloader:
            output = alexnet(item[0])
            target = item[1]
            # 使用交叉熵损失函数
            loss = nn.CrossEntropyLoss()(output, target)
            print('Epoch {}, Loss {}'.format(epoch + 1, loss))
            alexnet.zero_grad()
            loss.backward()
            optimizer.step()

    # 保存模型
    print(alexnet.state_dict())
    # 保存模型 只保存参数
    torch.save(alexnet.state_dict(), "./model/alexnet_for_cifar10.pth")


def load_trained_model():
    alexnet = models.alexnet()
    # 先改变输出维度
    fc_in_features = alexnet.classifier[6].in_features
    alexnet.classifier[6] = torch.nn.Linear(fc_in_features, 10)
    # 然后加载上面修改输出维度为10的训练之后的模型
    alexnet.load_state_dict(torch.load("./model/alexnet_for_cifar10.pth"))
    # 打印参数 参数太多了。。。
    # for parameter in alexnet.named_parameters():
    #     print(parameter)
    return alexnet


def test_train_result():
    alexnet = load_trained_model()

    # 准备验证数据 直接使用训练数据，这里给弄反了。那验证数据去训练，拿训练数据去测试。
    transform = transforms.Compose([transforms.Resize((224, 224)),  # 感觉这里RandomResize不是很好 Resize比较好。
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])  # 需要再研究下这个标准化的目的是啥 感觉标准化之后就看不清了
    cifar10_dataset = torchvision.datasets.CIFAR10(root="./data",
                                                   train=True,
                                                   transform=None,
                                                   target_transform=None,
                                                   download=True)
    cifar10_dataset_trans = torchvision.datasets.CIFAR10(root="./data",
                                                         train=True,
                                                         transform=transform,
                                                         target_transform=None,
                                                         download=True)

    # index = 30
    # print(cifar10_dataset_trans)
    # item = cifar10_dataset[index][0]
    # trans_item = cifar10_dataset_trans[index][0]
    # # 设置数据加载
    # item.show()
    # transforms.ToPILImage()(trans_item).show()
    #
    # # 验证结果
    # res = alexnet(trans_item.unsqueeze(0))
    # print_res(res)
    rightCount = 0
    for index in range(256):
        trans_item = cifar10_dataset_trans[index][0]
        res = alexnet(trans_item.unsqueeze(0))
        if get_res(res.argmax().item()) == get_res(cifar10_dataset_trans[index][1]):
            rightCount += 1
        print(get_res(res.argmax().item()) + "---" + get_res(cifar10_dataset_trans[index][1]))
    print("rate is:" + str(rightCount / 256))


def test_trained_model_by_local_img():
    alexnet = load_trained_model()
    img = Image.open('img/cat.jpg')
    transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = transform(img).unsqueeze(0)
    # 得到模型输出结果
    res = alexnet(input_tensor)
    # 打印类型
    print(get_res(res.argmax().item()))


# 打印结果
def get_res(index):
    cate = {
        0: "飞机",
        1: "骑车",
        2: "小鸟",
        3: "猫",
        4: "鹿",
        5: "狗",
        6: "狐狸",
        7: "马",
        8: "船",
        9: "卡车",
    }
    return cate.get(index, None)


if __name__ == '__main__':
    test_alexnet_model()
    # load_data()
    # get_modify_classifier()
    # start_train()
    # test_train_result()
    # test_trained_model_by_local_img()
