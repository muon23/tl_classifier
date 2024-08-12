from typing import List

import torch
from torch import nn


class ForwardNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int] = None,
            dropout: float = None,
            softmax: bool = True,
    ):
        super(ForwardNN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        dropout = nn.Dropout(p=dropout) if dropout else None

        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for layer in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[layer], hidden_dims[layer + 1]))
            # self.layers.append(nn.ReLU())
            # if dropout:
            #     self.layers.append(dropout)

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.softmax = nn.Softmax(dim=1) if softmax else None

        # self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # return self.layers(x)

        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def predict(self, x):
        x = self.forward(x)
        if self.softmax:
            x = self.softmax(x)
        return x
