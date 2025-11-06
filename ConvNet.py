import torch
import torch.nn as nn
'''
卷积的本质是一个滑动窗口操作,用一个小矩阵(卷积核/滤波器)在大矩阵(图像)上滑动,每次计算对应元素的加权和。
最大池化(Max Pooling) 是一个下采样操作,从局部区域中选择最大值作为代表,以减少空间尺寸和计算量,同时保留重要特征;每次池化后,后续卷积层能"看到"更大的原始图像区域
'''
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 第一个卷积层: 1个输入通道, 32个输出通道, 3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层: 32个输入通道, 64个输出通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
         # 最大池化层: 2x2池化核
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层1: 64*7*7 -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2: 128 -> 10 (假设有10个类别)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 卷积层1 + ReLU激活 
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # 展平张量以输入全连接层
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x