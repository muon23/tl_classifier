from typing import List

import torch
from torch import nn


class ForwardNN(nn.Module):
    """
    A generic feedforward neural network.

    This class defines a neural network architecture that consists of
    multiple fully connected layers. It allows for customizable hidden
    layer dimensions, dropout regularization, and optional softmax
    activation for the output layer.

    Attributes:
        input_dim (int): The number of input features.
        output_dim (int): The number of output classes.
        layers (nn.ModuleList): A list of layers in the neural network.
        dropout (nn.Dropout): Dropout layer for regularization, if specified.
        softmax (nn.Softmax): Softmax layer for output, if specified.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int] = None,
            dropout: float = None,
            softmax: bool = True,
    ):
        """
        Initializes the ForwardNN model.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of outputs.
            hidden_dims (List[int], optional): A list of integers specifying
                the number of neurons in each hidden layer. Defaults to
                [64, 32] if not provided.
            dropout (float, optional): The dropout probability for regularization.
                If None, dropout is not applied.
            softmax (bool): If True, applies softmax activation to the output layer.
        """

        super(ForwardNN, self).__init__()

        # Set default hidden dimensions if none are provided
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize the list of layers
        self.layers = nn.ModuleList()

        # Create a dropout layer if a dropout probability is specified
        self.dropout = nn.Dropout(p=dropout) if dropout else None

        # Add the the first hidden layers
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Add subsequent hidden layers
        for layer in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[layer], hidden_dims[layer + 1]))

        # Add the output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # initialize softmax layer if specified
        self.softmax = nn.Softmax(dim=1) if softmax else None

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """

        # Pass input through all layers except the last one
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))  # Apply ReLU activation
            if self.dropout:
                x = self.dropout(x)   # Apply dropout if specified

        # Pass through the output layer
        x = self.layers[-1](x)
        return x

    def predict(self, x):
        """
        Predicts the output for the given input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Predicted output tensor after applying softmax, if specified.
        """
        x = self.forward(x)  # Get the output from the forward pass
        if self.softmax:
            x = self.softmax(x)  # Apply softmax if specified
        return x