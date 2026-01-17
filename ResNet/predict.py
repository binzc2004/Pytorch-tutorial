import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import Residual,ResNet18  # 或 ResNet18 等

# -----------------------------
# 1. 设备
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. 类别（必须和训练时 ImageFolder 一致）
# -----------------------------
class_names = ['cat', 'dog']  # 注意顺序

# -----------------------------
# 3. 模型加载
# -----------------------------
model = ResNet18(Residual,in_channels=3,num_classes=len(class_names))
model.load_state_dict(torch.load("./ResNet/best_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 4. 预测用 transform（必须和 val/test 一致）
# -----------------------------
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 5. 单张图片预测函数
# -----------------------------
def predict_image(image_path):
    # 读取图片
    image = Image.open(image_path).convert("RGB")

    # 预处理
    image = predict_transform(image)

    # 增加 batch 维度: [3,224,224] -> [1,3,224,224]
    image = image.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()

    return pred_class, confidence, probs.cpu().numpy()

# -----------------------------
# 6. 测试
# -----------------------------
if __name__ == "__main__":
    img_path = "dog.png"   # 改成你的图片路径
    label, conf, all_probs = predict_image(img_path)

    print(f"预测结果: {label}")
    print(f"置信度: {conf:.4f}")
    print(f"各类别概率: {all_probs}")
