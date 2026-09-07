import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.activationFuncion = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha= 1e-4, beta=.75, k=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=.75, k=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=6)

        self.linear5 = nn.Linear(128, 100)
        self.linear6 = nn.Linear(100, 100)

    def forward(self, x):
        # print("input", x.abs().mean().item())
        # x = self.norm1(self.activationFuncion(self.conv1(x)))
        x = self.activationFuncion(self.conv1(x))
        x = self.pool(x)
        # print("conv1:", x.abs().mean().item())

        # x = self.norm2(self.activationFuncion(self.conv2(x)))
        x = self.activationFuncion(self.conv2(x))
        x = self.pool(x)
        # print("conv2:", x.abs().mean().item())

        x = self.activationFuncion(self.conv3(x))
        # print("conv3:", x.abs().mean().item())

        x = self.activationFuncion(self.conv4(x))
        # print("conv4:", x.abs().mean().item())

        x = torch.flatten(x, start_dim=1)

        x = self.activationFuncion(self.linear5(x))
        x = self.dropout(x)
        # print("linear1:", x.abs().mean().item())

        x = self.linear6(x)

        return x
