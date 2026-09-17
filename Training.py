import torch
import Network

def main():
    device = torch.device("cuda")

    AlexNet = Network.AlexNet().to(device)
    # AlexNet.load_state_dict(torch.load("Model.pt"))
    optim = torch.optim.SGD(AlexNet.parameters(), lr=0.1, momentum=0.5, weight_decay=0.0005)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    images, targets = torch.load(f"data/Processed_Images.pt")
    data = torch.utils.data.TensorDataset(images, targets)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True, num_workers=0)
    try:
        for epoch in range(100):
            for batch, (x, y) in enumerate(dataLoader):
                x = x.to(device)
                y = y.to(device)

                y_hat = AlexNet(x)
                loss = loss_fn(y_hat, y)

                optim.zero_grad()
                loss.backward()
                optim.step()
                # print([param.grad for param in AlexNet.parameters()])

                if(batch % 50 == 0):
                    print("loss: ", loss.item())
                if(batch % 500 == 0 and batch != 0):
                    if optim.param_groups[0]["lr"] <= 0.0001:
                        pass
                    else:
                        optim.param_groups[0]["lr"] *= 0.75

        torch.save(AlexNet.state_dict(), "Model.pt")

    except KeyboardInterrupt:
        torch.save(AlexNet.state_dict(), "Model.pt")


if __name__ == "__main__":
    main()