# %%
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from model import Tudui
from torch.utils.tensorboard import SummaryWriter
import time

# %%
train_data = torchvision.datasets.CIFAR10(root='./CIFAR', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./CIFAR', train=False, download=True, transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# %%
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# %%
tudui = Tudui()
tudui = tudui.cuda()
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# %%
total_train_step = 0
total_test_step = 0
epoch = 30

writer = SummaryWriter('logs')

start_time = time.time()

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i+1))

    tudui.train()
    for data in train_data_loader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("100次训练用时：{}".format(end_time-start_time))
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    total_test_loss = 0
    total_accuracy = 0

    tudui.eval()
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_accuracy += (outputs.argmax(1) == targets).sum().item()
            total_test_loss += loss.item()

    print("第{}轮训练的Loss为：{}".format(i+1, total_test_loss))
    print("第{}轮训练的准确率为：{}".format(i+1, total_accuracy/test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('accuracy', total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, './checkpoints/tudui_{}.pth'.format(i+1))
    print("-----第{}轮训练结束，模型已保存-----".format(i+1))


