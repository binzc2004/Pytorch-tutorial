import copy
import time

import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import AlexNet

def train_val_data_process():
    train_data= FashionMNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()])
                              )
    train_data,val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
    train_data_loader= Data.DataLoader(train_data, batch_size=32, shuffle=True,num_workers=2)
    val_data_loader= Data.DataLoader(val_data, batch_size=32, shuffle=False,num_workers=2)
    return train_data_loader,val_data_loader

def train_model_process(model,train_data_loader,val_data_loader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model=model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 初始化参数
        # 训练损失函数
        train_loss= 0.0
        # 训练准确率
        train_correct = 0
        # 验证损失函数
        val_loss = 0.0
        # 验证准确率
        val_correct = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        for step,(b_x, b_y) in enumerate(train_data_loader):
            # 将特征放到设备中
            b_x = b_x.to(device)
            # 将标签放到设备中
            b_y = b_y.to(device)
            model.train()
            # 前向传播过程，输出是特征向量
            output = model(b_x)

            pre_label = torch.argmax(output, 1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_correct += torch.sum(pre_label == b_y.data)
            train_num += b_x.size(0)

        for step,(b_x, b_y) in enumerate(val_data_loader):
            # 将特征放到设备中
            b_x = b_x.to(device)
            # 将标签放到设备中
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_label = torch.argmax(output, 1)
            loss = criterion(output, b_y)
            val_loss += loss.item() * b_x.size(0)
            val_correct += torch.sum(pre_label == b_y.data)
            val_num += b_x.size(0)

        train_loss_list.append(train_loss / train_num)
        train_acc_list.append(train_correct.double().item() / train_num)

        val_loss_list.append(val_loss / val_num)
        val_acc_list.append(val_correct.double().item() / val_num)
        print("train_loss:{:.4f} val_loss:{:.4f} train_acc:{:.4f} val_acc:{:.4f}".format(train_loss / train_num, val_loss / val_num, train_correct.double().item() / train_num, val_correct.double().item() / val_num))
        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, 'AlexNet/best_model.pth')
    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_list": train_loss_list,
            "val_loss_list": val_loss_list,
            "train_acc_list": train_acc_list,
            "val_acc_list": val_acc_list
        }
    )
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_acc_list"],'ro-',label="train_acc")
    plt.plot(train_process["epoch"], train_process["val_acc_list"],'bs-',label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_loss_list"],'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process["val_loss_list"],'bs-', label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('AlexNet/acc_loss.png')



if __name__ == '__main__':
    model = AlexNet()
    train_data_loader,val_data_loader = train_val_data_process()
    train_process = train_model_process(model,train_data_loader,val_data_loader,num_epochs=20)
    matplot_acc_loss(train_process)


