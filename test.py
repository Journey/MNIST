import torch
import torch.nn as nn
import config as config

def test(model: nn.Module):
    """测试模型"""
    # 推理模式， 影响 Dropout 和 BatchNorm 层的行为
    # Dropout: 停止丢弃神经元，使用完整网络进行推理
    # BatchNorm: 使用训练时计算的全局均值和方差，而不是当前批次的统计量
    model.eval()  # 设置模型为评估模式
    test_loss = 0.0
    correct = 0
    total = 0
    # 关闭tensor的自动微分机制， 不记录任何操作用于反向传播，节省内存和计算资源
    # 否则会为每个Tensor操作构建计算图： 存储每个中间结果 / 记录操作历史 / 维护梯度信息等。
    with torch.no_grad():  # 测试时不需要计算梯度
        for images, labels in config.test_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            test_loss += config.criterion(outputs, labels).item()
            total += labels.size(0) # tensor的第0维是batch size
            # 逐元素比较预测结果和真实标签，统计正确预测的数量， 转换成python 标量
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(config.test_loader)
    print(f'Test Set: 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy