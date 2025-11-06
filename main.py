from matplotlib import pyplot as plt
import torch
from ConvNet import ConvNet
from SimpleNN import SimpleNN
import config as config
from train import train
from test import test


torch.manual_seed(42)


model = SimpleNN().to(config.device)
# model = ConvNet().to(config.device) 

print("开始训练模型...")
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(config.num_epochs):
    print(f'\nEpoch {epoch + 1}/{config.num_epochs}')
    train_loss, train_accuracy = train(model)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_loss, test_accuracy = test(model)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# 配置中文字体 ✅
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 系统自带中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 绘制训练和测试的损失及准确率曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(range(1, config.num_epochs + 1), train_losses, 'b-', label='训练损失')
ax1.plot(range(1, config.num_epochs + 1), test_losses, 'r-', label='测试损失')
ax1.set_xlabel('轮次')
ax1.set_ylabel('损失')
ax1.set_title('训练和测试损失')
ax1.legend()
ax1.grid(True)

# 准确率曲线
ax2.plot(range(1, config.num_epochs + 1), train_accuracies, 'b-', label='训练准确率')
ax2.plot(range(1, config.num_epochs + 1), test_accuracies, 'r-', label='测试准确率')
ax2.set_xlabel('轮次')
ax2.set_ylabel('准确率 (%)')
ax2.set_title('训练和测试准确率')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print('训练曲线已保存为')










