import torch
import torchvision

trainData = torchvision.datasets.CIFAR100(root="data", train=False, download=True)

trainingData = torch.tensor(trainData.data)
trainingTargets = torch.tensor(trainData.targets)

data = trainingData
data = data.permute(0, 3, 1, 2)
data = data.float() / 255

torch.save((data, trainingTargets), "Testing_Processed_Images.pt")