Paper: ImageNet Classification with Deep Convolutional Neural Networks

This is an image classification network inspired by AlexNet but built for and trained on the CIFAR100 dataset.

Implementation:
the network is four convolutional layers followed by two fully connected layers, and implements dropout, pooling, and local response normalization similar to AlexNet's.
The network also uses ReLU as its activation function as inspired by AlexNet.
Training:
- Optimizer: Stochastic Gradient Descent
- Loss: Cross Entropy Loss
- Dataset: CIFAR 100
- Initial learning rate: 0.1
- Learning rate decay: *0.75 every 500 batches
- Minimum learning rate: 0.0001
- Batch size: 128
- Momentum: 0.5
- Weight decay: 0.0005

Results
Testing accuracy(Best): 42.10%
Testing accuracy(Current): 32.64%
Training accuracy: 40.88%

Notes:
Datasets and libraries are excluded to decrease project size.
Past models remain in models folder, however they may not function without reverting changes to network architecture.
This Network is still a work in progress, and my current target accuracy is 70%.
