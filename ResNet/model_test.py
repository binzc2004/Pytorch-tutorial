import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.utils.data as Data

from model import Residual,ResNet18


def test_data_process():
    ROOT_TEST= './data/test'
    test_transforms=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    test_data=ImageFolder(ROOT_TEST, transform=test_transforms)
    test_dataloader= Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    return test_dataloader

def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_correct = 0
    test_num = 0

    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            pre_label = torch.argmax(output, 1)

            test_correct += (pre_label == b_y).sum().item()
            test_num += b_x.size(0)

    print(f"test_acc: {test_correct / test_num:.4f}")

if __name__ == '__main__':
    test_dataloader = test_data_process()
    model = ResNet18(Residual, in_channels=3, num_classes=2)
    model.load_state_dict(torch.load('./ResNet/best_model.pth'))
    test_model_process(model, test_dataloader)
