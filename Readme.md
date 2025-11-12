# Playground: MNIST 手写数字识别

使用 PyTorch 实现的 MNIST 手写数字识别项目，包含简单全连接网络和卷积神经网络两种模型。

## 项目结构

```
.
├── main.py          # 主程序入口
├── train.py         # 训练函数
├── test.py          # 测试函数
├── SimpleNN.py      # 全连接神经网络
├── ConvNet.py       # 卷积神经网络
├── config.py        # 配置文件
└── data/            # MNIST 数据集
```

## 环境要求

- Python ~3.12
- PyTorch ^2.9.0
- torchvision ^0.24.0
- matplotlib ^3.10.7

## 安装依赖

```bash
poetry install
```

## 运行

```bash
make run
# 或
poetry run python -u main.py
```

## 模型说明

### SimpleNN (全连接网络)
- 输入层: 28×28 像素展平
- 隐藏层: 128 → 64 神经元
- 输出层: 10 个类别
- 激活函数: ReLU
- 正则化: Dropout (0.2)

### ConvNet (卷积神经网络)
- 卷积层: 1→32→64 通道
- 池化: 2×2 MaxPooling
- 全连接层: 3136 → 128 → 10
- Dropout: 0.25

## 训练配置

- 批次大小: 64
- 学习率: 0.001
- 训练轮数: 5
- 优化器: Adam
- 损失函数: CrossEntropyLoss

## 输出

- 模型权重: `model_epoch_*.pth`
- 训练曲线: `training_history.png`