import torch
import torch.nn as nn
import config as config

def test(model: nn.Module):
    """测试模型"""
    model.eval()  # 设置模型为评估模式
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 测试时不需要计算梯度
        for images, labels in config.test_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            test_loss += config.criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(config.test_loader)
    print(f'Test Set: 平均损失: {avg_loss:.4f}, 准确率: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy