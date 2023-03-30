import numpy as np
import tensorflow as tf

"""
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

class Network:
    def __init__(self, input_size, hidden_sizes, output_size = 1):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Define the layers of the neural network
        self.input_layer = tf.keras.layers.Input(shape=(input_size,))
        self.hidden_layers = []
        
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(tf.keras.layers.Dense(units=hidden_size, activation='relu'))
        
        self.output_layer = tf.keras.layers.Dense(units=output_size, activation='softmax')
        
        # Connect the layers to form the neural network
        activations = self.input_layer
        
        for hidden_layer in self.hidden_layers:
            activations = hidden_layer(activations)
            
        output_activations = self.output_layer(activations)
        
        # Define the model
        self.model = tf.keras.models.Model(inputs=self.input_layer, outputs=output_activations)
        
    def feedforward(self, inputs):
        # Perform feedforward operation using the trained model
        output_activations = self.model.predict(inputs)
        
        return output_activations
    
    def train(self, inputs, targets, learning_rate, num_epochs):
        # Compile the model with appropriate optimizer, loss function, and evaluation metric
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
        
        # Train the model on the input and target data for the specified number of epochs
        self.model.fit(inputs, targets, epochs=num_epochs)
        
    def save(self, filename):
        # Save the weights and biases of the model to a file
        self.model.save_weights(filename)
        
    def load(self, filename):
        # Load the weights and biases of the model from a file
        self.model.load_weights(filename)
