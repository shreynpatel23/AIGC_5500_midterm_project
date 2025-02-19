import torch.nn as nn
import torch
 
def initialize_model():
    """Creates and initializes the Feedforward Fully Connected Network."""
    
    # Define a sequential model with multiple layers
    model = nn.Sequential(
        nn.Flatten(),  # Flattens input tensor from (batch_size, 1, 28, 28) to (batch_size, 784) for MNIST-like data
        nn.Linear(784, 128),  # Fully connected layer from input size 784 to hidden layer of size 128
        nn.BatchNorm1d(128),  # Batch normalization to stabilize training and improve convergence
        nn.ReLU(),  # Activation function introducing non-linearity
        
        nn.Linear(128, 64),  # Second fully connected layer from 128 to 64 neurons
        nn.BatchNorm1d(64),  # Batch normalization for the second hidden layer
        nn.ReLU(),  # Activation function
 
        nn.Linear(64, 10)  # Output layer with 10 neurons (for 10 classes in classification)
    )
    
    # Weight Initialization
    # Using Kaiming Normal initialization for first hidden layer to improve training stability
    nn.init.kaiming_normal_(model[1].weight)  
    nn.init.constant_(model[1].bias, 0.0)  # Bias initialized to zero
    
    # Applying the same Kaiming Normal initialization to the second hidden layer
    nn.init.kaiming_normal_(model[4].weight)  
    nn.init.constant_(model[4].bias, 0.0)  
 
    # Using Xavier Uniform initialization for the output layer to maintain variance across layers
    nn.init.xavier_uniform_(model[7].weight)  
    nn.init.constant_(model[7].bias, 0.0)  
 
    return model  # Return the initialized model