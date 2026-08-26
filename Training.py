import torch
import Network

def main():
    device = torch.device("cuda")

    AlexNet = Network.AlexNet().to(device)
    optim = torch.optim.SGD(AlexNet.parameters(), lr=0.1, momentum=0.05, weight_decay=0.0000) #devide lr by 10 when the validation error rate stopps improving
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    images, targets = torch.load(f"data/Processed_Images.pt")
    data = torch.utils.data.TensorDataset(images, targets)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True, num_workers=0)

    for epoch in range(10):
        for batch, (x, y) in enumerate(dataLoader):
            x = x.to(device)
            y = y.to(device)

            y_hat = AlexNet(x)
            loss = loss_fn(y_hat, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if(batch % 50 == 0):
                print("loss: ", loss.item())

    torch.save(AlexNet.state_dict(), "Model.pt")

if __name__ == "__main__":
    main()