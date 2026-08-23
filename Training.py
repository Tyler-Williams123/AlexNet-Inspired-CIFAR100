import torch
import Network

def main():
    device = torch.device("cuda")

    AlexNet = Network.AlexNet().to(device)
    optim = torch.optim.SGD(AlexNet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005) #devide lr by 10 when the validation error rate stopps improving
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(2):
        for i in range(5):
            data = torch.load(f"data/Processed_Images_{i}.pt")
            data = torch.utils.data.TensorDataset(data)

            dataLoader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True, num_workers=3)

            for batch, (x, y) in enumerate(dataLoader):
                x = x.float() / 255
                x = x.to(device)
                y = y.to(device)

                y_hat = AlexNet(x)
                loss = loss_fn(y_hat, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                if(batch % 50 == 0):
                    print("loss: ", loss.item())

            del data

    torch.save(AlexNet.state_dict(), "Model.pt")

if __name__ == "__main__":
    main()