"""
description: todo
"""
import json
import numpy as np
from network import Network

# Load the trained network from the trained_network.weights file
network = Network(10, [16, 8], 3)
network.load('trained_network.weights')

# Load the projection data from the projection_data.json file
with open('projection_data.json', 'r') as f:
    projection_data = json.load(f)

# Extract the input data from the projection data
inputs = np.array(projection_data['inputs'])

# Define the number of days to project into the future
num_days = 30

# Perform feedforward operation for each day of the future projection
for i in range(num_days):
    # Use the trained network to predict the next day's data based on the previous data
    input_data = inputs[-10:].reshape(1, -1)
    output_data = network.feedforward(input_data)[0]
    
    # Append the predicted data to the inputs array for the next iteration
    inputs = np.vstack((inputs, output_data))
    
# Save the generated future projection to a file
projection_data['generated_data'] = inputs.tolist()

with open('projection_data.json', 'w') as f:
    json.dump(projection_data, f)
