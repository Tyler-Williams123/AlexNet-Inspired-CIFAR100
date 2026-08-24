import torch
import torch.nn.functional as F
import torchvision

trainData = torchvision.datasets.CIFAR100(root="data", train=False, download=True)

trainingData = torch.tensor(trainData.data)
trainingTargets = torch.tensor(trainData.targets)

data = trainingData
data = data.permute(0, 3, 1, 2)
data = data.float()

data = F.interpolate(data, (227, 227), mode="bilinear", align_corners=False)

data = data.to(dtype=torch.uint8)

torch.save((data, trainingTargets), f"Testing_Processed_Images.pt")