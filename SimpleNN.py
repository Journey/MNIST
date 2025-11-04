import config as config;
import torch.nn as nn

class SimpleNN(nn.Module):
    """全连接神经网络"""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        # 引入非线性判断能力： 将负值归0,保留正直
        self.relu = nn.ReLU()
        # 添加 Dropout 层，丢弃概率为 0.2, 防止过拟合, 依赖bernoulli分布
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 将多维张量展平为二维矩镇：保持批次纬度，展平其他纬度
        x = self.flatten(x)  # 展平输入图像

        x = self.relu(self.fc1(x))  # 第一层非线性变换
        x = self.dropout(x)

        x= self.relu(self.fc2(x))  # 第二层也需要激活函数
        x = self.dropout(x)

        # 输出层不需要激活
        x = self.fc3(x)
        return x