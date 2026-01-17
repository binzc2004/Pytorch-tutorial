import os
import time
import copy
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt

from model import Residual,ResNet18


# ================= 数据 =================
def train_val_data_process():
    # ImageNet 标准归一化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # -------- train（添加丰富的数据增强）--------
    ROOT_TRAIN = "./data/train"
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=8),  # 轻微位置扰动
        transforms.RandomHorizontalFlip(p=0.5),  # 猫狗 OK
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_data = ImageFolder(ROOT_TRAIN, transform=train_transforms)
    train_loader = Data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True
    )

    # -------- val（验证集仅做基础预处理，不增强）--------
    ROOT_VAL = "./data/val"
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_data = ImageFolder(ROOT_VAL, transform=val_transforms)
    val_loader = Data.DataLoader(
        val_data,
        batch_size=32,
        shuffle=False   # 验证集不打乱，保证评估稳定
    )

    return train_loader, val_loader


# ================= 训练 =================
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    checkpoint_path='./ResNet/checkpoint.pth'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ====== 断点相关变量 ======
    start_epoch = 0
    best_acc = 0.0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # ====== 自动加载 checkpoint ======
    if os.path.exists(checkpoint_path):
        print(f'>>> Resume from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        best_model_weights = model.state_dict()

        train_loss_list = checkpoint['train_loss_list']
        val_loss_list = checkpoint['val_loss_list']
        train_acc_list = checkpoint['train_acc_list']
        val_acc_list = checkpoint['val_acc_list']

    since = time.time()

    try:
        for epoch in range(start_epoch, num_epochs):
            print(f'\nEpoch {epoch}/{num_epochs - 1}')
            print('-' * 30)

            # ========== Train ==========
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_num = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)

                loss.backward()
                optimizer.step()

                pred = torch.argmax(output, 1)
                train_loss += loss.item() * x.size(0)
                train_correct += torch.sum(pred == y)
                train_num += x.size(0)

            train_loss /= train_num
            train_acc = train_correct.double().item() / train_num

            # ========== Val ==========
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_num = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    loss = criterion(output, y)

                    pred = torch.argmax(output, 1)
                    val_loss += loss.item() * x.size(0)
                    val_correct += torch.sum(pred == y)
                    val_num += x.size(0)

            val_loss /= val_num
            val_acc = val_correct.double().item() / val_num

            # ====== 记录 ======
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            print(
                f'train_loss:{train_loss:.4f} '
                f'val_loss:{val_loss:.4f} '
                f'train_acc:{train_acc:.4f} '
                f'val_acc:{val_acc:.4f}'
            )

            # ====== 保存 best model ======
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_weights = model.state_dict()

            # ====== 保存 checkpoint（每个 epoch 都保存） ======
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_loss_list': train_loss_list,
                'val_loss_list': val_loss_list,
                'train_acc_list': train_acc_list,
                'val_acc_list': val_acc_list
            }
            torch.save(checkpoint, checkpoint_path)

    except KeyboardInterrupt:
        print('\n>>> Training interrupted, checkpoint saved.')

    time_elapsed = time.time() - since
    torch.save(best_model_weights,"ResNet/best_model.pth")
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return pd.DataFrame({
        'epoch': range(len(train_loss_list)),
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
        'train_acc': train_acc_list,
        'val_acc': val_acc_list
    })

# ================= 画图 =================
def plot_acc_loss(df):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_acc'], label='train_acc')
    plt.plot(df['epoch'], df['val_acc'], label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.savefig('./ResNet/acc_loss.png')



# ================= 主入口 =================
if __name__ == '__main__':
    model = ResNet18(Residual,in_channels=3, num_classes=2)
    train_loader, val_loader = train_val_data_process()
    df = train_model(model, train_loader, val_loader, num_epochs=50)
    plot_acc_loss(df)


