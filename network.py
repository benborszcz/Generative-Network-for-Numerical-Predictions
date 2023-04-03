"""
network.py

utilize PyTorch

network.py
- holds the information pertaining to network structure and weights
- ability to save network
- ability to create networks of different sizes
- ability to train networks
- ability to feed-forward


What to pass as input?:
- previous data points x1000
- days since start of data x1
- day of the year (1-365) x1
- slope of a linear best fit analysis x1
- y-intercept of a linear best fit analysis x1
- ^1 slope of a quadratic best fit analysis x1
- ^2 slope of a quadratic best fit analysis x1
- y-intercept of a quadratic best fit analysis x1


Network Structure (variable just an example):
1007(input)
xxx
512(hidden)
xxx
256(hidden)
xxx
128(hidden)
xxx
1(ouput)

Activation Function: Leaky-ReLU

"""

import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Network, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = nn.functional.leaky_relu(layer(x))
        x = self.layers[-1](x)
        return x

    def save_network(self, path):
        torch.save(self.state_dict(), path)

    def load_network(self, path):
        self.load_state_dict(torch.load(path))