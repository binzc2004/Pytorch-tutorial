from torchvision import transforms
from torchvision.datasets import FashionMNIST
import numpy as np
import torch.utils.data as Data
train_data = FashionMNIST(root='./data',
                          train=True,
                          download=True,
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()])
                          )
train_loader= Data.DataLoader(train_data, batch_size=64, shuffle=True)