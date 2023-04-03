"""

MAIN GOALS:
- create a robust code base for generative future projections utilizing neural networks
- will be focused on date dependent data sets
- should be input size expandable dependent on the number of previous data points required
- an individual network will be created for each individual future projection needed
- networks will be trained on past data


FILES:

main.py
- runs the code

network.py
- holds the information pertaining to network structure and weights
- ability to save network
- ability to create networks of different sizes
- ability to train networks
- ability to feed-forward

training.py
- trains a network using training_data.json
- goes step by step and calls the backprop function in network

projection.py
- uses a trained network to complete a generative future projection of a numerical, date-dependent, data set
- continuously calls the feedforward function in network for each day of future projection needed

analyze.py
- graphs and displays the projections created in projection.py

training_data.json
- holds all of the known data for the network to train on

projection_data.json
- holds all of the genreated future projections


"""

import training
import projection
import analyze

def main():
    # Train the network
    trained_network = training.train_network("test_data.json")

    # Generate future projections
    projection_data = projection.generate_projections(trained_network)

    # Analyze and display the projections
    analyze.display_projections(training_file_name = "test_data.json")

if __name__ == "__main__":
    main()


