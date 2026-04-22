# tuning the original model for better ROI
# last update by ZJUHITL Team, Dec 9, 2021

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision import models
from torchvision.models import ResNet
import numpy as np
import matplotlib.pyplot as plt
import os
import utils


data_dir = './data/ROI'
save_path = os.path.join(data_dir+'/model/', 'checkpoints')
train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                                 transform=transforms.Compose(
                                                     [
                                                         transforms.Resize((640,640)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))
                                                     ]))

val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'),
                                               transform=transforms.Compose(
                                                     [
                                                         transforms.Resize((640,640)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))
                                                     ]))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=1)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=1)

#类别名称
class_names = train_dataset.classes
print('class_names:{}'.format(class_names))

#训练设备 CPU/GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("train_device:{}".format(device.type))

# 随机显示一个batch
#plt.figure()
#utils.imshow(next(iter(train_dataloader)))
#plt.show()

# -------------------------模型选择，优化方法， 学习率策略----------------------
# model = models.squeezenet1_1(pretrained=True)

# model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048,2)
#模型迁移到CPU/GPU
model = model.to(device)

#定义损失函数
loss_fc = nn.CrossEntropyLoss()

#选择性优化方法
optimizer = optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)

#学习率调整策略
#每7个epoch调整一次
exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer,step_size=10,gamma=0.5) #step_size

# ----------------训练过程-----------------
num_epochs = 20

for epoch in range(num_epochs):

    running_loss = 0.0
    exp_lr_scheduler.step()

    for i, sample_batch in enumerate(train_dataloader):
        inputs = sample_batch[0]
        labels = sample_batch[1]

        model.train()

        # GPU/CPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # foward
        outputs = model(inputs)

        # loss
        loss = loss_fc(outputs, labels)

        # loss求导，反向
        loss.backward()

        # 优化
        optimizer.step()

        #
        running_loss += loss.item()

        # 測試
        if i % 20 == 19:
            correct = 0
            total = 0
            model.eval()
            for images_test, labels_test in val_dataloader:
                images_test = images_test.to(device)
                labels_test = labels_test.to(device)

                outputs_test = model(images_test)
                _, prediction = torch.max(outputs_test, 1)
                correct += (torch.sum((prediction == labels_test))).item()
               # print(prediction, labels_test, correct)
                total += labels_test.size(0)
            print('[{}, {}] running_loss = {:.5f} accurcay = {:.5f}'.format(epoch + 1, i + 1, running_loss / 20,
                                                                        correct / total))
            running_loss = 0.0

        if i % 10 == 9:
            print('[{}, {}] loss={:.5f}'.format(epoch+1, i+1, running_loss / 10))
            running_loss = 0.0

    if epoch % 10 ==9:    
        print('Checkpoint ', epoch + 1, ' saved !')
        save_prefix = 'roi'
        save_path = os.path.join('./model/checkpoints/', f'{save_prefix}{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)

print('training finish !')
torch.save(model.state_dict(), './model/model_resnet_roi.pth')
