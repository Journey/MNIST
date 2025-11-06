import torch
from ConvNet import ConvNet
from SimpleNN import SimpleNN
import config as config
from train import train
from test import test


torch.manual_seed(42)


# model = SimpleNN().to(config.device)
model = ConvNet().to(config.device) 

print("开始训练模型...")
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(config.num_epochs):
    print(f'\nEpoch {epoch + 1}/{config.num_epochs}')
    train_loss, train_accuracy = train(model)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_loss, test_accuracy = test(model, config.device, config.test_loader, config.criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')










