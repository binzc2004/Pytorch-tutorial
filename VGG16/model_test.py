import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
import torch.utils.data as Data

from model import VGG16


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             download=True,
                             transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()])
                             )
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
    model = VGG16()
    model.load_state_dict(torch.load('./VGG16/best_model.pth'))
    test_model_process(model, test_dataloader)
