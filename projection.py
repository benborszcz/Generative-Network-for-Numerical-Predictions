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

def generate_projections(trained_network, num_days=200, run_num = 0):
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print("Using device:", device)

    network = trained_network.to(device)

    with open("training_data.json", "r") as file:
        data_points = json.load(file)

    projection_data = []
    config = configparser.ConfigParser()
    config.read("config.ini")

    sliding_window_size = int(config.get("config", "sliding_window_size"))
    #sliding_window_size = len(data_points)-100
    input_size = sliding_window_size + 7

    for i in range(num_days):
        current_data = data_points[-sliding_window_size:]
        input_data = [point["value"] for point in current_data]
        additional_params = calculate_additional_parameters(current_data, current_data[-1]["timestamp"] + 86400)
        input_data.extend(additional_params)

        inputs = torch.tensor(input_data, dtype=torch.float32).to(device)
        output = network(inputs)
        projection_data.append(output.item())

        data_points.append({"timestamp": current_data[-1]["timestamp"] + 86400, "value": output.item()})

    with open(f"projection_data_{run_num}.json", "w") as file:
        json.dump(projection_data, file)

    return projection_data