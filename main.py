import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from SimpleNN import SimpleNN
import config as config
from train import train
from test import test


torch.manual_seed(42)


model = SimpleNN().to(config.device)

print("开始训练模型...")
train_losses, train_accuracies = [], []

for epoch in range(config.num_epochs):
    print(f'\nEpoch {epoch + 1}/{config.num_epochs}')
    train_loss, train_accuracy = train(model)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)









