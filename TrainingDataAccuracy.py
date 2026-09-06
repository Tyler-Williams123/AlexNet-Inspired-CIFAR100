import torch
import Network

def main():
    device = torch.device("cuda")

    AlexNet = Network.AlexNet().to(device)
    AlexNet.load_state_dict(torch.load("Model.pt"))

    images, targets = torch.load(f"data/Processed_Images.pt")
    data = torch.utils.data.TensorDataset(images, targets)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True, num_workers=0)

    AlexNet.eval()
    with torch.no_grad():
        for x, y in dataLoader:
            x = x.to(device)
            y = y.to(device)

            guess = torch.argmax(AlexNet(x), dim=1)
            correct = sum((guess == y))
            print(correct / y.shape[0])


if __name__ == "__main__":
    main()