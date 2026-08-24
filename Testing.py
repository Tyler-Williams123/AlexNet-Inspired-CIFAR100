import Network

import torch

dataset = torch.load("data/Testing_Processed_Images.pt")
AlexNet = Network.AlexNet()
AlexNet.load_state_dict(torch.load("model.pt"))

x, y = dataset

AlexNet.eval()

with torch.no_grad:
    correct = (torch.argmax(AlexNet(x), dim=1) == y).sum()

    percentage = correct / y.shape[0]
    print(percentage)