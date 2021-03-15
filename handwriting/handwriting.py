#!/usr/bin/env python3
import torch
import torchvision
import matplotlib.pyplot as plt

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=True, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=1000, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307, ), (0.3081, ))
                               ])),
    batch_size=64, shuffle=True)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title(f'Ground Truth: {example_targets[i]}')
    plt.xticks([])
    plt.yticks([])
# plt.show()

w1 = torch.rand(16, 784)
b1 = torch.rand(16)
layer_1 = torch.zeros(16)
w2 = torch.rand(16, 16)
b2 = torch.rand(16)
layer_2 = torch.zeros(16)
w3 = torch.rand(16, 16)
b3 = torch.rand(16)
output = torch.zeros(10)

for p in range(1000):
    sigmoid = torch.nn.ReLU()
    for i in range(16):
        weightedInput = example_data[p][0].flatten().dot(w1[i])
        layer_1[i] = sigmoid(weightedInput.add(b1[i]))

    for i in range(16):
        weightedInput = layer_1.dot(w2[i])
        layer_2[i] = sigmoid(weightedInput.add(b2[i]))

    for i in range(10):
        weightedInput = layer_2.dot(w3[i])
        output[i] = sigmoid(weightedInput.add(b3[i]))
    print("Output", torch.topk(output, 1).indices, " - Target",
          example_targets[p])
