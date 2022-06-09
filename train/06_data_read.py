import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    # 返回数据集的大小
    def __len__(self):
        return self.data_tensor.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


if __name__ == '__main__':
    # data_tensor = torch.randn(10, 3)
    # print(data_tensor)
    # target_tensor = torch.randint(2, (10,))  # 标签是0或1
    # print(target_tensor)
    # my_dataset = MyDataset(data_tensor, target_tensor)
    # print("my_dataset size:", len(my_dataset))
    # print("my_dataset[0]:", my_dataset[1])
    #
    # tensor_dataloader = DataLoader(dataset=my_dataset, batch_size=3, shuffle=False, num_workers=1)
    # for data, target in tensor_dataloader:
    #     print(data, target)

    mnist_dataset = torchvision.datasets.MNIST(root="./data",
                                               train=False,
                                               transform=None,
                                               target_transform=None,
                                               download=True)
    mnist_dataset_list = list(mnist_dataset)
    # print(mnist_dataset_list)
    # 如何可视化直接先是一个pil
    # display(mnist_dataset_list[0][0])
    pic = mnist_dataset_list[0][0]
    pic.show()
    print(pic)
    label = mnist_dataset_list[0][1]
    print("Image label is:", label)
