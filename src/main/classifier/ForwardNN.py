from typing import List

from torch import nn


class ForwardNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int] = None,
            dropout: float = None,
    ):
        super(ForwardNN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        dropout = nn.Dropout(p=dropout) if dropout else None

        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for layer in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[layer], hidden_dims[layer + 1]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(dropout)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
