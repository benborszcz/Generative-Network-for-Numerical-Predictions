import json
import numpy as np
import random
import time

def generate_financial_data(days, noise_factor=0.05, trend_factor=0.001):
    time_points = np.arange(days)
    trend = np.sin(trend_factor * time_points) * 100
    random_walk = np.random.choice([-1, 1], days) * np.random.rand(days) * 10
    random_walk[0] = 0  # Starting at 0
    random_walk = np.cumsum(random_walk)
    noise = np.random.normal(0, noise_factor * max(trend), days)
    return trend + random_walk + noise

def main():
    num_days = 1000
    financial_data = generate_financial_data(num_days)

    unix_start_time = int(time.time()) - num_days * 86400  # 86400 seconds in a day
    data_points = [{"timestamp": unix_start_time + 86400 * i, "value": financial_data[i]} for i in range(num_days)]

    with open("training_data.json", "w") as file:
        json.dump(data_points, file)

if __name__ == "__main__":
    main()