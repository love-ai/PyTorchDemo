import torch
import torchvision.datasets
from PIL.Image import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


# 模型为图像分类模型 先提供出来
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        # conv1 输出的特征图为222*222大小
        self.fc = nn.Linear(16 * 222 * 222, 10)

    def forward(self, input):
        x = self.conv1(input)
        # 进入全连接层之前，先将特征图铺平
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def train():
    # 设置训练硬件配置 尽量GPU 这里只能cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化模型
    model = MyCNN().to(device)

    # 数据读取
    transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                   train=False,
                                                   transform=transform,
                                                   target_transform=None,
                                                   download=True)
    # 设置数据加载
    dataLoader = DataLoader(dataset=cifar10_dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=2)
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

    # 训练
    for epoch in range(3):
        for item in dataLoader:
            img = item.to(device)
            output = model(img[0])
            target = img[1]
            # 使用交叉熵损失函数, 数据、模型、损失函数，有GPU的话都要放到GPU上。
            loss = nn.CrossEntropyLoss().to(device)(output, target)
            print('Epoch {}, Loss {}'.format(epoch + 1, loss))
            model.zero_grad()
            loss.backward()
            optimizer.step()
    # 保存默认
    # torch.save(model.state_dict(), "./model/mycnn_for_cifar10.pth")


def test():
    # 先创建网络结构
    model = MyCNN()
    # 加载保存的参数
    model.load_state_dict(torch.load("./model/mycnn_for_cifar10.pth"))
    img = Image.open('img/cat.jpg')
    transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = transform(img).unsqueeze(0)
    # 得到模型输出结果
    res = model(input_tensor)
    # 打印类型
    print(res.argmax())


if __name__ == '__main__':
    train()
    # test()
