import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from SimpleNN import SimpleNN
import config as config

# 设置随机种子以确保结果可复现: 相同的种子会产生相同的随机数序列 => 伪随机
torch.manual_seed(42)

def train(model: nn.Module):
    """训练模型"""
    model.train()  # 设置模型为训练模式
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(config.num_epochs):
        for batch_idx, (data, target) in enumerate(config.train_loader):
            data, target = data.to(config.device), target.to(config.device)

            optimizer.zero_grad()  # 清除之前的梯度信息
            output = model(data)   # 前向传播
            loss = config.criterion(output, target)  # 计算损失
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新模型参数

            # 统计信息
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{config.num_epochs}], Step [{batch_idx+1}/{len(config.train_loader)}], Loss: {loss.item():.6f}')
    # 计算平均损失和准确率
    avg_loss = train_loss / len(config.train_loader)
    accuracy = 100. * correct / total
    print(f'训练集: 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy