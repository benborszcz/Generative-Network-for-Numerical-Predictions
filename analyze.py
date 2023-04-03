"""
description: todo
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def display_projections(training_file_name = "training_data.json", run_num=0):
    # Load the training data
    with open("training_data.json", "r") as file:
        training_data = json.load(file)

    training_values = [point["value"] for point in training_data]

    # Load the projection data
    with open(f"projection_data_{run_num}.json", "r") as file:
        data = json.load(file)

    # Calculate the running total for training and projection data
    running_total_training = np.cumsum(training_values)
    running_total_projection = np.cumsum(data) + running_total_training[-1]

    # Plot the training data
    plt.plot(running_total_training, color='blue', label='Training Data')

    # Plot the projections
    projection_start = len(training_values)
    projection_end = projection_start + len(data)
    plt.plot(range(projection_start, projection_end), running_total_projection, color='red', label='Projections')

    plt.xlabel("Days")
    plt.ylabel("Running Total")
    plt.title("Generative Future Projections")
    plt.legend()
    plt.show()