import torch
import torch.nn.functional as F
import torchvision

trainData = torchvision.datasets.CIFAR100(root="data", train=True, download=True)

trainingData = torch.tensor(trainData.data)
trainingTargets = torch.tensor(trainData.targets)

for i in range(5):
    data = trainingData[i * 10_000 : (i + 1) * 10_000]
    data = data.permute(0, 3, 1, 2)
    data = data.float()

    data = F.interpolate(data, (224, 224), mode="bilinear", align_corners=False)

    data = data.to(dtype=torch.uint8)

    torch.save((data, trainingTargets[i * 10_000 : (i + 1) * 10_000]), f"Processed_Images_{i}.pt")