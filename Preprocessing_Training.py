import torch
import torchvision

trainData = torchvision.datasets.CIFAR100(root="data", train=True, download=True)

trainingData = torch.tensor(trainData.data)
trainingTargets = torch.tensor(trainData.targets)

trainingData = trainingData.permute(0, 3, 1, 2)
trainingData = trainingData.float() / 255

torch.save((trainingData, trainingTargets), "Processed_Images.pt")