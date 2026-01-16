import os
from PIL import Image
import torch
from torchvision import transforms

# ================== 配置 ==================
TRAIN_DIR = "./data/train"
IMG_SIZE = (224, 224)
# ==========================================

def calculate_mean_std(train_dir):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()   # 一定要先转 Tensor
    ])

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_pixels = 0

    for cls_name in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img)  # [3, H, W]

                channel_sum += img.sum(dim=[1, 2])
                channel_squared_sum += (img ** 2).sum(dim=[1, 2])
                num_pixels += img.shape[1] * img.shape[2]

            except Exception as e:
                print(f"跳过图片: {img_path}")

    mean = channel_sum / num_pixels
    std = (channel_squared_sum / num_pixels - mean ** 2).sqrt()

    return mean, std


if __name__ == "__main__":
    mean, std = calculate_mean_std(TRAIN_DIR)

    print("====== 训练集 Mean ======")
    print(mean.tolist())

    print("====== 训练集 Std ======")
    print(std.tolist())
