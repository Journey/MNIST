from matplotlib import transforms
from torchvision import datasets, transforms
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64  
learning_rate = 0.001
num_epochs = 5
# 损失函数: 测量模型预测的概率分布与真实标签分布之间的差异
criterion = torch.nn.CrossEntropyLoss() # 适用与多分类任务

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])


train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)