"""
description: todo
"""

import json
import numpy as np
from network import Network

# Load the training data from the training_data.json file
with open('training_data.json', 'r') as f:
    training_data = json.load(f)

# Extract the input and target data from the training data
inputs = np.array(training_data['inputs'])
targets = np.array(training_data['targets'])

# Define the size of the input and output layers based on the shape of the input and target data
input_size = inputs.shape[1]
output_size = targets.shape[1]

# Define the size of the hidden layers
hidden_sizes = [16, 8]

# Create a neural network object
network = Network(input_size, hidden_sizes, output_size)

# Define the learning rate and number of epochs for training
learning_rate = 0.001
num_epochs = 1000

# Train the network using backpropagation and the training data
for i in range(num_epochs):
    network.backpropagation(inputs, targets, learning_rate)

# Save the trained network to a file
network.save('trained_network.weights')
