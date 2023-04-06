# Generative Future Projections with Neural Networks

This project aims to create a robust codebase for generative future projections utilizing neural networks. The primary focus is on date-dependent datasets, and the code allows for input size expandability based on the number of previous data points required. An individual network is created for each future projection needed, and the networks are trained on past data.

## Files

### main.py

- Runs the code.

### network.py

- Holds the information pertaining to network structure and weights.
- Allows for saving network states.
- Creates networks of different sizes.
- Trains networks.
- Performs feed-forward operations.

### training.py

- Trains a network using `training_data.json`.
- Goes step by step and calls the backprop function in the network.

### projection.py

- Uses a trained network to complete a generative future projection of a numerical, date-dependent dataset.
- Continuously calls the feedforward function in the network for each day of future projection needed.

### analyze.py

- Graphs and displays the projections created in `projection.py`.

### training_data.json

- Holds all the known data for the network to train on.

### projection_data.json

- Holds all the generated future projections.

### config.ini

- Holds the configuration parameters for the neural network, such as sliding window size, hidden layer sizes, and epochs.

## How to Run

1. Ensure that all dependencies are installed. This project requires PyTorch, NumPy, and Matplotlib.
2. Prepare your training data in JSON format and save it as `training_data.json`.
3. Configure the neural network parameters in `config.ini`.
4. Run `main.py` to train the network, generate future projections, and display the projections.

## Example Usage

```python
import training
import projection
import analyze

def main():
    # Train the network
    trained_network = training.train_network()

    # Generate future projections
    projection_data = projection.generate_projections(trained_network)

    # Analyze and display the projections
    analyze.display_projections()

if __name__ == "__main__":
    main()
```

## Note

This code is designed to handle numerical, date-dependent datasets. The input to the neural network includes previous data points, days since the start of data, the day of the year, and parameters derived from linear and quadratic best-fit analyses. The network structure and activation functions can be customized as needed.
