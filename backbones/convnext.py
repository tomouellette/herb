import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple


class ConvNeXtV2(nn.Module):
    """A ConvNeXt V2 model

    Parameters
    ----------
    in_chans : int
        Number of input channels
    n_classes : int
        Number of classes for classification head
    depths : Tuple[int]
        Number of blocks at each stage
    dims : Tuple[int]
        Feature dimension/channel size at each stage
    drop_path_rate : float
        Stochastic depth rate
    head_init_scale : float
        Init scaling value for classifier weights and biases

    References
    ----------
    1. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
       Autoencoders". Woo et al. (2023) https://arxiv.org/pdf/2301.00808
    2. github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnext.py
    """

    def __init__(
        self,
        in_chans: int = 3,
        n_classes: int = 1000,
        depths: Tuple[int] = (3, 3, 9, 3),
        dims: Tuple[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.,
        head_init_scale: float = 1.
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.out_dim = dims[-1]
        self.in_chans = in_chans
        self.n_classes = n_classes

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(OrderedDict([
            ("in_conv", nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)),
            ("in_norm", LayerNorm(dims[0], eps=1e-6, channel_pos="first"))
        ])))

        for i in range(3):
            downsample_layer = [
                (f"downsample_norm{i}", LayerNorm(
                    dims[i],
                    eps=1e-6,
                    channel_pos="first"
                )),
                (f"downsample_conv{i}", nn.Conv2d(
                    dims[i],
                    dims[i+1],
                    kernel_size=2,
                    stride=2
                )),
            ]
            self.downsample_layers.append(
                nn.Sequential(OrderedDict(downsample_layer))
            )

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        current_stage = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                OrderedDict([
                    (f"convnext_block{j}", ConvNeXtBlock(
                        in_chans=dims[i],
                        drop_path=dp_rates[current_stage + j]
                    )) for j in range(depths[i])
                ])
            )
            self.stages.append(stage)
            current_stage += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], n_classes)

        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def forward_embed(self, x: Tensor) -> Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_embed(x)
        x = self.head(x)
        return x

    def save(self, path: str, kind: str = None):
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        for k, v in [
            ("in_chans", self.in_chans),
            ("depths", self.depths),
            ("dims", self.dims),
            ("out_dim", self.out_dim),
            ("n_classes", self.n_classes),
        ]:
            dtype = torch.int64
            if isinstance(v, (list, tuple)):
                state_dict[k] = torch.tensor(v, dtype=dtype)
            else:
                state_dict[k] = torch.tensor([v], dtype=dtype)

        if kind == "torch" or path.endswith((".pth", ".pt")):
            if not path.endswith(".pth"):
                path = path + ".pth"

            torch.save(self.state_dict(), path)
        elif kind == "safetensors" or path.endswith(".safetensors"):
            from safetensors.torch import save_file
            if not path.endswith(".safetensors"):
                path = path + ".safetensors"

            save_file(state_dict, path)


class DropPath(nn.Module):
    """Stochastic depth by dropping paths/samples

    References
    ----------
    1. github.com/rwightman/pytorch-image-models/blob/main/timm/layers/drop.py
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class GRN(nn.Module):
    """ Global response normalization to avoid feature collapse

    References
    ----------
    1. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
       Autoencoders". Woo et al. (2023) https://arxiv.org/pdf/2301.00808
    2. github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    """

    def __init__(self, in_chans: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, in_chans))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, in_chans))

    def forward(self, x: Tensor) -> Tensor:
        x_global = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        x_norm = x_global / (x_global.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * x_norm) + self.beta + x


class LayerNorm(nn.Module):
    """Channel-first or channel-last layer normalization

    References
    ----------
    1. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
       Autoencoders". Woo et al. (2023) https://arxiv.org/pdf/2301.00808
    2. github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        channel_pos: str = "last"
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.channel_pos = channel_pos
        if self.channel_pos not in ["last", "first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: Tensor) -> Tensor:
        if self.channel_pos == "last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.channel_pos == "first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def move_channel(x: Tensor, channel_pos: str) -> Tensor:
    if channel_pos == "last":
        return x.permute(0, 2, 3, 1)
    elif channel_pos == "first":
        return x.permute(0, 3, 1, 2)
    else:
        raise NotImplementedError


class ConvNeXtBlock(nn.Module):
    """A ConvNeXt block following version 2 implementation

    Parameters
    ----------
    in_chans : int
        Number of input channels
    expansion_factor : int
        Expansion factor for the pointwise convolution block
    drop_path : float
        Drop path rate

    References
    ----------
    1. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
       Autoencoders". Woo et al. (2023) https://arxiv.org/pdf/2301.00808
    2. github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    """

    def __init__(
        self,
        in_chans: int,
        expansion_factor: int = 4,
        drop_path: float = 0.
    ):
        super().__init__()
        self.depthwise_convolution = nn.Conv2d(
            in_chans,
            in_chans,
            kernel_size=7,
            padding=3,
            groups=in_chans,
        )

        self.pointwise_convolutions = nn.Sequential(OrderedDict([
            ('pw_norm', nn.LayerNorm(in_chans, eps=1e-6)),
            ('pw_linear1', nn.Linear(in_chans, expansion_factor * in_chans)),
            ('pw_gelu', nn.GELU()),
            ('pw_grn', GRN(expansion_factor * in_chans)),
            ('pw_linear2', nn.Linear(expansion_factor * in_chans, in_chans)),
        ]))

        self.drop_path = nn.Identity()
        if drop_path > 0.:
            self.drop_path = DropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.depthwise_convolution(x)
        x = move_channel(x, "last")
        x = self.pointwise_convolutions(x)
        x = move_channel(x, "first")
        x = input + self.drop_path(x)
        return x


def convnext_nano(
    channels: int = 3,
    image_size: int = None,
    drop_path_rate: float = 0.,
    n_classes: int = 1000,
):
    return ConvNeXtV2(
        in_chans=channels,
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 200],
        head_init_scale=0.001,
        n_classes=n_classes,
    )


def convnext_micro(
    channels: int = 3,
    image_size: int = None,
    drop_path_rate: float = 0.,
    n_classes: int = 1000,
):
    return ConvNeXtV2(
        in_chans=channels,
        depths=[2, 2, 6, 2],
        dims=[48, 96, 192, 384],
        head_init_scale=0.001,
        n_classes=n_classes,
    )


def convnext_tiny(
    channels: int = 3,
    image_size: int = None,
    drop_path_rate: float = 0.,
    n_classes: int = 1000,
):
    return ConvNeXtV2(
        in_chans=channels,
        depths=[2, 2, 8, 2],
        dims=[64, 128, 256, 512],
        head_init_scale=0.001,
        n_classes=n_classes,
    )


def convnext_small(
    channels: int = 3,
    image_size: int = None,
    drop_path_rate: float = 0.,
    n_classes: int = 1000,
):
    return ConvNeXtV2(
        in_chans=channels,
        depths=[3, 3, 9, 3],
        dims=[80, 160, 320, 640],
        head_init_scale=0.001,
        n_classes=n_classes,
    )


def convnext_base(
    channels: int = 3,
    image_size: int = None,
    drop_path_rate: float = 0.,
    n_classes: int = 1000,
):
    return ConvNeXtV2(
        in_chans=channels,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        head_init_scale=0.001,
        n_classes=n_classes,
    )


def convnext_large(
    channels: int = 3,
    image_size: int = None,
    drop_path_rate: float = 0.,
    n_classes: int = 1000,
):
    return ConvNeXtV2(
        in_chans=channels,
        depths=[3, 3, 28, 3],
        dims=[192, 384, 768, 1536],
        head_init_scale=0.001,
        n_classes=n_classes,
    )


if __name__ == "__main__":
    prefix = "[INFO | convnext ]"

    print(f"{prefix} Checking ConvNeXt forward pass.")

    model = ConvNeXtV2(
        in_chans=3,
        n_classes=4321,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768)
    )

    x = torch.randn(1, 3, 224, 224)

    embed = model.forward_embed(x)
    assert embed.shape == (1, 768), \
        f"{prefix} Embed failed. Shape is {embed.shape}"

    out = model(x)
    assert out.shape == (1, 4321), \
        f"{prefix} Head failed. Shape is {out.shape}"

    nano = convnext_nano()
    micro = convnext_micro()
    tiny = convnext_tiny()
    small = convnext_small()
    base = convnext_base()
    large = convnext_large()

    nano.save("backbones/candle_convnext/convnext.safetensors")

    def _n_parameters(model):
        model.head = nn.Identity()
        return sum(p.numel() for p in model.parameters())

    print(f"{prefix} Nano model has {_n_parameters(nano)} parameters.")
    print(f"{prefix} Micro model has {_n_parameters(micro)} parameters.")
    print(f"{prefix} Tiny model has {_n_parameters(tiny)} parameters.")
    print(f"{prefix} Small model has {_n_parameters(small)} parameters.")
    print(f"{prefix} Base model has {_n_parameters(base)} parameters.")
    print(f"{prefix} Large model has {_n_parameters(large)} parameters.")

    print(f"{prefix} Basic ConvNeXt checks passed.")

    import time
    x = torch.randn(1, 3, 224, 224)

    def runtime(model, name):
        start = time.time()
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        end = time.time()
        mean = (end - start) / 10
        print(
            f"{prefix} Inference for {name} is {mean:.3f} images/second (CPU)."
        )

    runtime(nano, "nano")
    runtime(micro, "micro")
    runtime(tiny, "tiny")
    runtime(small, "small")
    runtime(base, "base")
    runtime(large, "large")
