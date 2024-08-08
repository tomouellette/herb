import torch
from torch import nn
from torch import Tensor
from typing import Union
from collections import OrderedDict


class ViT(nn.Module):
    """Vision transformer with multi-head self-attention and registers

    Parameters
    ----------
    image_size : Union[int, tuple]
        Size of input image
    patch_size : Union[int, tuple]
        Size of image patch
    n_classes : int
        Number of classes
    dim : int
        Dimension of token embeddings
    depth : int
        Number of transformer blocks
    heads : int
        Number of attention heads
    mlp_dim : int
        Dimension of feedforward network
    pool : str
        Type of pooling, either 'cls' (class token) or 'mean' (mean pooling)
    channels : int
        Number of input channels
    dim_head : int
        Dimension of each attention head
    dropout : float
        Dropout rate
    emb_dropout : float
        Dropout rate for token embeddings

    References
    ----------
    1. "An Image is Worth 16x16 Words: Transformers for Image Recognition
       at Scale". Dosovitskiy et al. (2020). https://arxiv.org/abs/2010.11929
    2. "Vision Transformers need Registers". Darcet et al. (2024).
       https://arxiv.org/pdf/2309.16588
    3. github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    """

    def __init__(
        self,
        image_size: Union[int, tuple] = 224,
        patch_size: Union[int, tuple] = 16,
        n_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 8,
        mlp_dim: int = 512,
        pool: str = 'mean',
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        n_registers: int = 8,
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_height = image_width = image_size
        elif isinstance(image_size, tuple):
            image_height, image_width = image_size
        else:
            raise ValueError('image_size must be an int or a tuple')

        if isinstance(patch_size, int):
            patch_height = patch_width = patch_size
        elif isinstance(patch_size, tuple):
            patch_height, patch_width = patch_size
        else:
            raise ValueError('patch_size must be an int or a tuple')

        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError('Image size must be divisible by patch size')

        n_patches = (image_height // patch_height) \
            * (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width

        self.patch_height = patch_height
        self.patch_width = patch_width

        if pool not in {'cls', 'mean'}:
            raise ValueError(
                'pool must be either cls (class token) or mean (mean pooling)'
            )

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.n_registers = n_registers
        self.register_tokens = nn.Parameter(torch.randn(n_registers, dim))

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout
        )

        self.pool = pool
        self.head = nn.Linear(dim, n_classes)

        self.image_height = image_height
        self.image_width = image_width
        self.in_chans = channels
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        self.n_registers = n_registers

    def forward_embed(self, x: Tensor):
        b, c, h, w = x.shape
        p1, p2 = self.patch_height, self.patch_width

        ph = h // p1
        pw = w // p2

        x = x.reshape(b, c, ph, p1, pw, p2)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(b, ph * pw, p1 * p2 * c)

        x = self.to_patch_embedding(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(ph * pw + 1)]
        x = self.dropout(x)

        r = self.register_tokens.expand(b, -1, -1)

        x = torch.cat((r, x), dim=1)
        x = self.transformer(x)
        x = x[:, self.n_registers:, :]

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return x

    def forward(self, img):
        x = self.forward_embed(img)
        x = self.head(x)
        return x

    def save(self, path: str, kind: str = None):
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        for k, v in [
            ("image_height", self.image_height),
            ("image_width", self.image_width),
            ("in_chans", self.in_chans),
            ("patch_height", self.patch_height),
            ("patch_width", self.patch_width),
            ("dim", self.dim),
            ("depth", self.depth),
            ("heads", self.heads),
            ("mlp_dim", self.mlp_dim),
            ("dim_head", self.dim_head),
            ("n_registers", self.n_registers),
        ]:
            dtype = torch.int64
            if isinstance(v, (list, tuple)):
                state_dict[k] = torch.tensor(v, dtype=dtype)
            else:
                state_dict[k] = torch.tensor([v], dtype=dtype)

        state_dict["mean_pool"] = torch.tensor(
            [self.pool == "mean"], dtype=torch.int64
        )

        if kind == "torch" or path.endswith((".pth", ".pt")):
            if not path.endswith(".pth"):
                path = path + ".pth"

            torch.save(self.state_dict(), path)
        elif kind == "safetensors" or path.endswith(".safetensors"):
            from safetensors.torch import save_file
            if not path.endswith(".safetensors"):
                path = path + ".safetensors"

            save_file(state_dict, path)


class Attention(nn.Module):
    """Multi-head self-attention

    Reference
    ---------
    1. github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        b, n, h = q.shape
        d = h // self.heads

        q = q.reshape(b, n, self.heads, d)
        q = q.permute(0, 2, 1, 3)

        k = k.reshape(b, n, self.heads, d)
        k = k.permute(0, 2, 1, 3)

        v = v.reshape(b, n, self.heads, d)
        v = v.permute(0, 2, 1, 3)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(b, -1, self.heads * self.dim_head)

        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer with multi-head self-attention

    Reference
    ---------
    1. github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.
    ):
        super().__init__()
        self.depth = depth
        self.norm = nn.LayerNorm(dim)
        for i in range(depth):
            self.add_module(
                f"attention_{i}",
                Attention(
                    dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout
                )
            )

            self.add_module(
                f"feedforward_{i}",
                nn.Sequential(OrderedDict([
                    ("norm", nn.LayerNorm(dim)),
                    ("linear1", nn.Linear(dim, mlp_dim)),
                    ("gelu", nn.GELU()),
                    ("dropout1", nn.Dropout(dropout)),
                    ("linear2", nn.Linear(mlp_dim, dim)),
                    ("dropout2", nn.Dropout(dropout)),
                ]))
            )

    def forward(self, x):
        for i in range(self.depth):
            x = getattr(self, f"attention_{i}")(x) + x
            x = getattr(self, f"feedforward_{i}")(x) + x

        return self.norm(x)


def vit_nano(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000,
) -> ViT:
    return ViT(
        image_size=image_size,
        channels=channels,
        patch_size=patch_size,
        n_classes=n_classes,
        dim=196,
        depth=5,
        heads=2,
        mlp_dim=4 * 196,
        dim_head=196 // 2,
        dropout=0.,
        emb_dropout=0.,
        n_registers=8,
    )


def vit_micro(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000,
) -> ViT:
    return ViT(
        image_size=image_size,
        channels=channels,
        patch_size=patch_size,
        n_classes=n_classes,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=4 * 192,
        dim_head=192 // 3,
        dropout=0.,
        emb_dropout=0.,
        n_registers=8,
    )


def vit_tiny(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000,
) -> ViT:
    return ViT(
        image_size=image_size,
        channels=channels,
        patch_size=patch_size,
        n_classes=n_classes,
        dim=256,
        depth=12,
        heads=4,
        mlp_dim=4 * 256,
        dim_head=256 // 4,
        dropout=0.,
        emb_dropout=0.,
        n_registers=8,
    )


def vit_small(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000,
) -> ViT:
    return ViT(
        image_size=image_size,
        channels=channels,
        patch_size=patch_size,
        n_classes=n_classes,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=4 * 384,
        dropout=0.,
        dim_head=384 // 6,
        emb_dropout=0.,
        n_registers=8,
    )


def vit_base(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000,
) -> ViT:
    return ViT(
        image_size=image_size,
        channels=channels,
        patch_size=patch_size,
        n_classes=n_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=4 * 768,
        dropout=0.,
        dim_head=768 // 12,
        emb_dropout=0.,
        n_registers=8,
    )


def vit_large(
    image_size: int = 224,
    channels: int = 3,
    patch_size: int = 16,
    n_classes: int = 1000,
) -> ViT:
    return ViT(
        image_size=image_size,
        channels=channels,
        patch_size=patch_size,
        n_classes=n_classes,
        dim=1024,
        depth=16,
        heads=16,
        mlp_dim=4 * 1024,
        dropout=0.,
        dim_head=1024 // 16,
        emb_dropout=0.,
        n_registers=8,
    )


if __name__ == "__main__":
    prefix = "[INFO | vit ]"

    print(f"{prefix} Checking ViT forward pass.")

    model = ViT(
        image_size=224,
        channels=3,
        patch_size=32,
        n_classes=4321,
        dim=1234,
        n_registers=7,
    )

    x = torch.randn(1, 3, 224, 224)

    embed = model.forward_embed(x)
    assert embed.shape == (1, 1234), \
        f"{prefix} Embed failed. Shape is {embed.shape}"

    out = model(x)
    assert out.shape == (1, 4321), \
        f"{prefix} Head failed. Shape is {out.shape}"

    nano = vit_nano()
    micro = vit_micro()
    tiny = vit_tiny()
    small = vit_small()
    base = vit_base()
    large = vit_large()

    nano.save("backbones/candle_vit/vit.safetensors")

    def _n_parameters(model):
        model.head = nn.Identity()
        return sum(p.numel() for p in model.parameters())

    print(f"{prefix} Nano model has {_n_parameters(nano)} parameters.")
    print(f"{prefix} Micro model has {_n_parameters(micro)} parameters.")
    print(f"{prefix} Tiny model has {_n_parameters(tiny)} parameters.")
    print(f"{prefix} Small model has {_n_parameters(small)} parameters.")
    print(f"{prefix} Base model has {_n_parameters(base)} parameters.")
    print(f"{prefix} Large model has {_n_parameters(large)} parameters.")

    print(f"{prefix} Basic ViT checks passed.")

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
