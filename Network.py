import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,):
        super.__init__()
        self.activationFuncion = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha= 1e-4, beta=.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, groups=2)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, groups=2, padding=1)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, groups=2, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.linear6 = nn.Linear(9216, 4096)
        self.linear7 = nn.Linear(4096, 4096)

        self.linear8 = nn.Linear(4096, 1000)

        self.softMax = nn.Softmax()



    def forward(self, x):
        x = self.norm1(self.activationFuncion(self.conv1(x)))
        x = self.pool1(x)

        x = self.norm2(self.activationFuncion(self.conv2(x)))
        x = self.pool2(x)

        x = self.activationFuncion(self.conv3(x))

        x = self.activationFuncion(self.conv4(x))

        x = self.activationFuncion(self.conv5(x))
        x = self.pool5(x)

        x = torch.flatten(x, start_dim=1)

        x = self.activationFuncion(self.linear6(x))
        x = self.dropout(x)

        x = self.activationFuncion(self.linear7(x))
        x = self.dropout(x)

        x = self.linear8(x)
        x = self.softMax(x)
        return x
