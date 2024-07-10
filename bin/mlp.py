import torch
import torch.nn as nn


class MLP(nn.Module):
    """A multi-layer (fully-connected) perceptron

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    hidden_layers : list
        List of integers specifying the number of units in each hidden layer
    output_dim : int
        Dimension of output features
    dropout_rate : float
        Dropout rate, defaults to 0
    activation : torch.nn.modules.activation
        Pytorch activation function, defaults to nn.ReLU
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list,
        output_dim: int,
        dropout_rate: float = 0,
        activation: nn.Module = nn.ReLU,
    ):
        super(MLP, self).__init__()
        if len(hidden_layers) == 0 or not isinstance(hidden_layers, list):
            raise ValueError("Hidden layers must be a list of integers.")

        if not hasattr(nn.modules.activation, str(activation()).split('(')[0]):
            raise ValueError("Activation must be nn.modules.activation.")

        if (dropout_rate < 0) | (dropout_rate > 1):
            raise ValueError("Dropout must be between 0 and 1.")

        layers = [nn.Linear(input_dim, hidden_layers[0]), activation()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            layers.append(activation())

        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.layers(x)
