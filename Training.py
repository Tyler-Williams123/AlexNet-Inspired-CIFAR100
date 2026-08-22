import torch
import torchvision.transforms as transforms
import Network

AlexNet = Network.AlexNet()
optim = torch.optim.SGD(AlexNet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005) #devide lr by 10 when the validation error rate stopps improving
loss_fn = torch.nn.CrossEntropyLoss()

for i in range(5):
    data = torch.load(f"Processed_Images_{i}.pt")

    dataLoader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=3)

    for batch, (x, y) in enumerate(dataLoader):
        x = x.float() / 255

        y_hat = AlexNet(y)
        loss = loss_fn(y_hat, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if(batch % 50 == 0):
            print("loss: " + loss)