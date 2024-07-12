import torch
from torch import nn
from typing import Union


class MLPMixer(nn.Module):
    """An all MLP architecture for patch-based image modeling

    Parameters
    ----------
    img_size : Union[tuple, int]
        Size of the image. If tuple, it must be (height, width)
    in_chans : int
        Number of channels of the input image
    patch_size : int
        Size of the patch
    dim : int
        Embed patches to a vector of size dim
    depth : int
        Number of mixer layers, defaults to 12
    n_classes : int
        Number of classes, defaults to 100
    expansion_factor : int
        Expansion factor for the channels mixer
    expansion_factor_token : float
        Expansion factor token that scales down dimension of patches mixer
    dropout : float
        Dropout rate, defaults to 0.

    References
    ----------
    1. I. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai,
       T. Unterthiner, J. Yung, A. Steiner, D. Keysers, J. Uszkoreit,
       M. Lucic, A. Dosovitskiy. "MLP-Mixer: An all-MLP Architecture
       for Vision". https://arxiv.org/abs/2105.01601 2021.
    2. https://github.com/lucidrains/mlp-mixer-pytorch: Mixer layers adapted
       from lucidrains concise mlp-mixer implementation.
    """

    def __init__(
        self,
        img_size: Union[tuple, int] = 224,
        in_chans: int = 3,
        patch_size: int = 16,
        dim: int = 512,
        depth: int = 12,
        n_classes: int = 100,
        expansion_factor: int = 4,
        expansion_factor_token: float = 0.5,
        dropout: float = 0.
    ):
        super().__init__()

        if isinstance(img_size, tuple):
            image_h, image_w = img_size
        else:
            image_h, image_w = (img_size, img_size)

        if not (image_h % patch_size) == 0 or not (image_w % patch_size) == 0:
            raise ValueError('Image must be divisible by patch size')

        self.patch_size = patch_size
        self.num_patches = (image_h // patch_size) * (image_w // patch_size)

        self.patch_embed = nn.Linear((patch_size ** 2) * in_chans, dim)

        self.mixer_layers = nn.ModuleList([])
        for _ in range(depth):
            channels_mixer = [
                nn.LayerNorm(dim),
                nn.Conv1d(
                    self.num_patches,
                    int(self.num_patches * expansion_factor),
                    kernel_size=1
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    int(self.num_patches * expansion_factor),
                    self.num_patches,
                    kernel_size=1
                ),
                nn.Dropout(dropout)
            ]

            tokens_mixer = [
                nn.LayerNorm(dim),
                nn.Linear(dim, int(dim * expansion_factor_token)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(dim * expansion_factor_token), dim),
                nn.Dropout(dropout)
            ]

            self.mixer_layers.append(
                nn.Sequential(*channels_mixer, *tokens_mixer)
            )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.AvgPool2d(kernel_size=(self.num_patches, 1)),
            nn.Linear(dim, n_classes)
        )

    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape

        ps = self.patch_size
        ph = h // ps
        pw = w // ps

        x = x.reshape(-1, c, ph, ps, pw, ps)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(-1, ph * pw, c * ps * ps)

        x = self.patch_embed(x)

        for mixer in self.mixer_layers:
            x = mixer(x) + x

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_embed(x)
        x = self.head(x).squeeze(1)
        return x


def mlp_mixer_tiny(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000
) -> MLPMixer:
    return MLPMixer(
        img_size=image_size,
        in_chans=channels,
        patch_size=16,
        dim=768,
        depth=6,
        n_classes=n_classes,
        expansion_factor=4,
        expansion_factor_token=0.5,
        dropout=0.
    )


def mlp_mixer_small(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000
) -> MLPMixer:
    return MLPMixer(
        img_size=image_size,
        in_chans=channels,
        patch_size=16,
        dim=1024,
        depth=8,
        n_classes=n_classes,
        expansion_factor=4,
        expansion_factor_token=0.5,
        dropout=0.
    )


def mlp_mixer_base(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000
) -> MLPMixer:
    return MLPMixer(
        img_size=image_size,
        in_chans=channels,
        patch_size=16,
        dim=1280,
        depth=12,
        n_classes=n_classes,
        expansion_factor=4,
        expansion_factor_token=0.5,
        dropout=0.
    )


def mlp_mixer_large(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000
) -> MLPMixer:
    return MLPMixer(
        img_size=image_size,
        in_chans=channels,
        patch_size=16,
        dim=1536,
        depth=24,
        n_classes=n_classes,
        expansion_factor=4,
        expansion_factor_token=0.5,
        dropout=0.
    )


if __name__ == "__main__":
    prefix = "[INFO | mlp_mixer ]"

    print(f"{prefix} Checking MLP-Mixer forward pass.")

    model = MLPMixer(patch_size=32, n_classes=4321, dim=1234)
    x = torch.randn(1, 3, 224, 224)

    embed = model.forward_embed(x)
    assert embed.shape == (1, 49, 1234), \
        f"{prefix} Embed failed. Shape is {embed.shape}"

    out = model(x)
    assert out.shape == (1, 4321), \
        f"{prefix} Head failed. Shape is {out.shape}"

    tiny = mlp_mixer_tiny()
    small = mlp_mixer_small()
    base = mlp_mixer_base()
    large = mlp_mixer_large()

    def _n_parameters(model):
        return sum(p.numel() for p in model.parameters())

    print(f"{prefix} Tiny model has {_n_parameters(tiny)} parameters.")
    print(f"{prefix} Small model has {_n_parameters(small)} parameters.")
    print(f"{prefix} Base model has {_n_parameters(base)} parameters.")
    print(f"{prefix} Large model has {_n_parameters(large)} parameters.")

    print(f"{prefix} Basic MLP-Mixer checks passed.")
