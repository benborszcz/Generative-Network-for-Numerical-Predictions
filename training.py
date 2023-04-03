"""
description: todo
"""
import torch
import torch.cuda as cuda
import json
import numpy as np
import configparser
from network import Network

def calculate_additional_parameters(data, current_timestamp):
    days_since_start = (current_timestamp - data[0]["timestamp"]) // 86400
    day_of_year = days_since_start % 365

    values = np.array([point["value"] for point in data])

    # Calculate the cumulative sum of values
    cum_values = np.cumsum(values)

    time = np.arange(len(cum_values))
    slope, intercept = np.polyfit(time, cum_values, 1)
    coef2, coef1, intercept_quad = np.polyfit(time, cum_values, 2)

    return [
        days_since_start,
        day_of_year,
        slope,
        intercept,
        coef1,
        coef2,
        intercept_quad
    ]

def train_network(file_name="training_data.json"):

    # Check if GPU is available
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print("Using device:", device)

    config = configparser.ConfigParser()
    config.read("config.ini")

    with open(file_name, "r") as file:
        data_points = json.load(file)


    sliding_window_size = int(config.get("config", "sliding_window_size"))
    #sliding_window_size = len(data_points)-100
    input_size = sliding_window_size + 7
    hidden_sizes = list(map(int, config.get("config", "hidden_layer_sizes").split(", ")))
    output_size = 1
    network = Network(input_size, hidden_sizes, output_size).to(device)

    epochs = int(config.get("config", "epochs"))
    learning_rate = 0.001
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(epoch)
        for i in range(sliding_window_size, len(data_points)):
            print(f"[{i - sliding_window_size}-{i}]")
            input_data = [point["value"] for point in data_points[i - sliding_window_size:i]]
            additional_params = calculate_additional_parameters(data_points[:i], data_points[i]["timestamp"])
            input_data.extend(additional_params)

            inputs = torch.tensor(input_data, dtype=torch.float32).to(device)
            target = torch.tensor([data_points[i]["value"]], dtype=torch.float32).to(device)  # Wrap the target in a list

            output = network(inputs)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    network.save_network("trained_network.pth")
    return network