import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class MLP(nn.Module):
    """A multi-layer (fully-connected) perceptron

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    hidden_dim : list
        List of integers specifying the number of units in each hidden layer
    output_dim : int
        Dimension of output features
    dropout_rate : float
        Dropout rate, defaults to 0
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: List[int],
        output_dim: int,
        dropout_rate: float = 0,
        batch_norm_depth: int = 2,
    ):
        super(MLP, self).__init__()
        assert len(hidden_dim) > 0, "Hidden layers must be list of integers."
        assert 0 <= dropout_rate <= 1, "Dropout must be between 0 and 1."

        layers = [nn.Linear(input_dim, hidden_dim[0]), nn.GELU()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))

        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))

            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            if i - 1 > batch_norm_depth:
                layers.append(nn.BatchNorm1d(hidden_dim[i+1]))

            if i < len(hidden_dim) - 2:
                layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim[-1], output_dim)
        )

    def forward_embed(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_embed(x.flatten(1))
        x = self.head(x)
        return x


def mlp_nano(
    image_size: int = 224,
    channels: int = 3,
    n_classes: int = 1000,
    dropout_rate: float = 0.,
) -> MLP:
    return MLP(
        input_dim=image_size * image_size * channels,
        hidden_dim=[32],
        output_dim=n_classes,
        dropout_rate=dropout_rate
    )


def mlp_micro(
    image_size: int = 224,
    channels: int = 3,
    n_classes: int = 1000,
    dropout_rate: float = 0.,
) -> MLP:
    return MLP(
        input_dim=image_size * image_size * channels,
        hidden_dim=[64, 64],
        output_dim=n_classes,
        dropout_rate=dropout_rate
    )


def mlp_tiny(
    image_size: int = 224,
    channels: int = 3,
    n_classes: int = 1000,
    dropout_rate: float = 0.,
) -> MLP:
    return MLP(
        input_dim=image_size * image_size * channels,
        hidden_dim=[128, 128, 128],
        output_dim=n_classes,
        dropout_rate=dropout_rate
    )


def mlp_small(
    image_size: int = 224,
    channels: int = 3,
    n_classes: int = 1000,
    dropout_rate: float = 0.,
) -> MLP:
    return MLP(
        input_dim=image_size * image_size * channels,
        hidden_dim=[256, 256, 256, 256],
        output_dim=n_classes,
        dropout_rate=dropout_rate
    )


def mlp_base(
    image_size: int = 224,
    channels: int = 3,
    n_classes: int = 1000,
    dropout_rate: float = 0.,
) -> MLP:
    return MLP(
        input_dim=image_size * image_size * channels,
        hidden_dim=[512, 512, 512, 512, 512],
        output_dim=n_classes,
        dropout_rate=dropout_rate
    )


def mlp_large(
    image_size: int = 224,
    channels: int = 3,
    n_classes: int = 1000,
    dropout_rate: float = 0.,
) -> MLP:
    return MLP(
        input_dim=image_size * image_size * channels,
        hidden_dim=[2048, 2048, 2048, 2048, 2048, 2048],
        output_dim=n_classes,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    prefix = "[INFO | mlp ]"

    print(f"{prefix} Checking MLP forward pass.")

    model = MLP(
        input_dim=224*224*3,
        hidden_dim=[512, 512],
        output_dim=4321,
        dropout_rate=0.1
    )

    x = torch.randn(2, 3, 224, 224)

    embed = model.forward_embed(x.flatten(1))
    assert embed.shape == (2, 512), \
        f"{prefix} Embed failed. Shape is {embed.shape}"

    out = model(x)
    assert out.shape == (2, 4321), \
        f"{prefix} Head failed. Shape is {out.shape}"

    nano = mlp_nano()
    micro = mlp_micro()
    tiny = mlp_tiny()
    small = mlp_small()
    base = mlp_base()
    large = mlp_large()

    def _n_parameters(model):
        model.head = nn.Identity()
        return sum(p.numel() for p in model.parameters())

    print(f"{prefix} Nano model has {_n_parameters(nano)} parameters.")
    print(f"{prefix} Micro model has {_n_parameters(micro)} parameters.")
    print(f"{prefix} Tiny model has {_n_parameters(tiny)} parameters.")
    print(f"{prefix} Small model has {_n_parameters(small)} parameters.")
    print(f"{prefix} Base model has {_n_parameters(base)} parameters.")
    print(f"{prefix} Large model has {_n_parameters(large)} parameters.")

    print(f"{prefix} Basic MLP checks passed.")
