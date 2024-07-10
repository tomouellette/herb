import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional


class NaViT(nn.Module):
    """Dimension-agnostic vision transformer backbone following NaViT

    Parameters
    ----------
    max_image_size : int
        Maximum allowable image size for variable dimension inputs
    n_classes : int
        Number of classes for classification
    patch_size : int
        Size of the patch patchs
    dim : int
        Dimension of the patch embeddings
    depth : int
        Number of transformer layers
    heads : int
        Number of attention heads
    mlp_dim : int
        Dimension of the feedforward network
    channels : int
        Number of channels in the input images
    dim_head : int
        Dimension of the attention heads
    dropout : float
        Dropout rate for the transformer layers
    emb_dropout : float
        Dropout rate for the embedding layer
    token_drop : float
        Probability of dropping a patch from a packed image
    packed_sequence_length : int
        Maximum sequence length for packing multiple images in a batch.
        This must be longer than the maximum number of patchs generated
        by the largest image in the batch.

    References
    ----------
    1. Dehgani et al. "Patch n’ Pack: NaViT, a Vision Transformer for any
       Aspect Ratio and Resolution". NeurIPS (2023).
    2. github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/na_vit.py:
       This implementation trys to follow the lucidrains implementation of
       NaViT. It's been decomposed into pack, pad, embed, and pool steps.
    """

    def __init__(
        self,
        max_image_size: int = 512,
        n_classes: int = 10,
        patch_size: int = 8,
        dim: int = 10,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 64,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        token_drop: float = 0.0,
        packed_sequence_length: Optional[int] = None,
    ):
        super().__init__()
        assert 0.0 <= token_drop < 1.0
        assert max_image_size % patch_size == 0, \
            "Max image size must be divisible by patch size."

        if packed_sequence_length is None:
            packed_sequence_length = (max_image_size / patch_size) ** 2

        th = tw = max_image_size // patch_size
        patch_dimension = channels * (patch_size ** 2)

        assert tw * th <= packed_sequence_length, \
            f"Packed sequence length must be >= max # of patches {tw * th}."

        self.max_image_size = max_image_size
        self.channels = channels
        self.patch_size = patch_size
        self.token_drop = token_drop
        self.packed_sequence_length = packed_sequence_length

        self.patch_embedding = nn.Sequential(
            LayerNorm(patch_dimension),
            nn.Linear(patch_dimension, dim),
            LayerNorm(dim),
        )

        self.pos_embed_height = nn.Parameter(torch.randn(th, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(tw, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout
        )

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)

        self.head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, n_classes, bias=False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_pack(self, images: List[Tensor]) -> Tuple[Tensor, ...]:
        device = self.device

        # NOTE: Images first need to be formatted into batches that have total
        # number of patchs that do not exceed the specified max packed sequence
        # length. Once we reach an image that adds patchs that surpasses the
        # max, we just start a new batch and then resume appending so that we
        # don't lose any images. We also perform asserts here so we catch
        # errors before padding.

        total_patchs, batches, batch = 0, [], []
        for image in images:
            assert image.ndim == 3, "Images must be 3-dimensional"

            assert isinstance(image, Tensor), \
                "All entries in `images` must be a torch.Tensor"

            assert max(image.shape) <= self.max_image_size, \
                f"All images must be smaller than {self.max_image_size}" + \
                " in height or width"

            c, h, w = image.shape
            assert c == self.channels, \
                f"All images must have {self.channels} channels"

            th, tw = h / self.patch_size, w / self.patch_size

            assert th.is_integer(), \
                f"Image height ({h}) must be divisible by " + \
                f"patch size ({self.patch_size})"

            assert tw.is_integer(), \
                f"Image width ({w}) must be divisible by " + \
                f"patch size ({self.patch_size})"

            n_patchs = int(th * tw * (1 - self.token_drop))

            if n_patchs + total_patchs > self.packed_sequence_length:
                batches.append(batch)
                total_patchs, batch = 0, []

            batch.append(image)
            total_patchs += n_patchs

        if len(batch) > 0:
            batches.append(batch)

        # NOTE: Given we now have batches of images with total patchs <= max
        # packed sequence length, we now need to reshape the images into
        # flattened sequences of patchs with patch height and width equal to
        # the specified patch size. We also need to store the corresponding
        # positional embeddings for each patch in the sequence.

        num_images = []
        packed_sequences, packed_positions, packed_ids = [], [], []
        for batch in batches:
            num_images.append(len(batch))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device=device, dtype=torch.long)

            for image_id, image in enumerate(batch):
                c, h, w = image.shape

                th, tw = h // self.patch_size, w // self.patch_size
                n_patchs = th * tw

                position = torch.stack(torch.meshgrid((
                    torch.arange(th, device=device),
                    torch.arange(tw, device=device)
                ), indexing='ij'), dim=-1)

                position = position.view(th * tw, 2)

                sequence = (
                    image
                    .view(c, th, self.patch_size, tw, self.patch_size)
                    .permute(1, 3, 0, 2, 4)
                    .reshape(n_patchs, c * self.patch_size * self.patch_size)
                )

                if self.token_drop > 0.0:
                    num_keep = max(1, int(n_patchs * (1 - self.token_drop)))
                    keep_indices = torch.randn(
                        (int(n_patchs),),
                        device=device
                    ).topk(num_keep, dim=-1).indices

                    sequence = sequence[keep_indices]
                    position = position[keep_indices]

                    n_patchs = num_keep

                image_ids = F.pad(image_ids, (0, n_patchs), value=image_id)

                sequences.append(sequence)
                positions.append(position)

            packed_ids.append(image_ids)
            packed_sequences.append(torch.cat(sequences, dim=0))
            packed_positions.append(torch.cat(positions, dim=0))

        return num_images, packed_ids, packed_sequences, packed_positions

    def forward_pad(
        self,
        num_images: List[Tensor],
        packed_ids: List[Tensor],
        packed_sequences: List[Tensor],
        packed_positions: List[Tensor],
    ) -> Tuple[Tensor, ...]:
        device = self.device

        sequence_lengths = torch.tensor(
            [sequence.shape[-2] for sequence in packed_sequences],
            device=device,
            dtype=torch.long
        )

        # NOTE: The packed mask is a matrix with same shape as padded packed
        # sequences where each row is a boolean mask indicating which
        # patchs are real and which are padded.

        max_patch_arange = torch.arange(sequence_lengths.amax().item())
        packed_mask = max_patch_arange[None, :] < sequence_lengths[:, None]

        # NOTE: This is the source of the actual pad_sequence implementation.
        # Keeping it here just in case I want to re-implement later if I write
        # an inference model in rust or C++. Might be fun to re-write in Candle
        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/PackedSequence.cpp#L205

        packed_ids = pad_sequence(packed_ids, batch_first=True)
        packed_sequences = pad_sequence(packed_sequences, batch_first=True)
        packed_positions = pad_sequence(packed_positions, batch_first=True)

        num_images = torch.tensor(num_images, device=device, dtype=torch.long)

        return (
            num_images,
            packed_ids,
            packed_sequences,
            packed_positions,
            packed_mask
        )

    def forward_tokens(self, images: List[Tensor]) -> Tuple[Tensor, ...]:
        (
            n_images,
            image_ids,
            sequences,
            positions,
            padding_mask
        ) = self.forward_pad(
            *self.forward_pack(images)
        )

        attention_mask = \
            image_ids[:, None, :, None] == image_ids[:, None, None, :]

        attention_mask = attention_mask & padding_mask[:, None, None, :]

        x = self.patch_embedding(sequences)

        h_indices, w_indices = positions.unbind(dim=-1)
        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]

        x = x + h_pos + w_pos

        x = self.dropout(x)
        x = self.transformer(x, attention_mask=attention_mask)

        return x, n_images, image_ids, padding_mask

    def forward_pool(
        self,
        x: Tensor,
        n_images: Tensor,
        image_ids: Tensor,
        padding_mask: Tensor
    ) -> Tensor:
        max_queries = n_images.amax().item()
        queries = self.attn_pool_queries.repeat(x.shape[0], max_queries, 1)

        image_id_arange = torch.arange(max_queries)
        attention_pool_mask = image_id_arange[:, None] == image_ids[:, None, :]
        attention_pool_mask = attention_pool_mask & padding_mask[:, None, :]
        attention_pool_mask = attention_pool_mask[:, None, :, :]

        x = self.attn_pool(
            queries,
            context=x,
            attention_mask=attention_pool_mask
        ) + queries

        x = x.view(-1, x.shape[-1])

        is_images = image_id_arange < n_images[:, None]
        is_images = is_images.flatten()
        x = x[is_images]

        return x

    def forward_embed(self, images: List[Tensor]) -> Tensor:
        x, n_images, image_ids, padding_mask = self.forward_tokens(images)
        x = self.forward_pool(x, n_images, image_ids, padding_mask)
        return x

    def forward(self, images: List[Tensor], training: bool = True) -> Tensor:
        if not training:
            self.token_drop = 0.0

        x = self.forward_embed(images)
        x = self.head(x)

        return x


class LayerNorm(nn.Module):
    """Layer normalization without bias

    References
    ----------
    1. github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/na_vit.py#L75
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(nn.Module):
    """Query-key normalization using RMS normalization

    References
    ----------
    1. github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/na_vit.py#L86
    """

    def __init__(self, heads: int, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


class Attention(nn.Module):
    """Multi-head self-attention with optional masking

    References
    ----------
    1. github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/na_vit.py#L108
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head

        self.norm = LayerNorm(dim)
        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: Tensor,
        context: Tensor = None,
        mask: Tensor = None,
        attention_mask: Tensor = None
    ) -> Tensor:
        batch, n = x.shape[0:2]

        x = self.norm(x)
        kv_input = x if context is None else context

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = q.view(batch, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(batch, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(batch, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if mask is not None:
            mask = mask[:, None, None, :]
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if attention_mask is not None:
            dots = dots.masked_fill(
                ~attention_mask,
                -torch.finfo(dots.dtype).max
            )

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(
            batch, -1, self.heads * self.dim_head
        )

        return self.to_out(out)


class Transformer(nn.Module):
    """Vision transformer with masked multi-head attention

    References
    ----------
    1. github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/na_vit.py#L162
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
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout
                ),
                nn.Sequential(
                    LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        attention_mask: Tensor = None
    ) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x, mask=mask, attention_mask=attention_mask) + x
            x = ff(x) + x

        return self.norm(x)


if __name__ == "__main__":
    prefix = "[INFO | navit ]"

    print(f"{prefix} Checking NaViT forward pass.")

    model = NaViT(
        patch_size=8,
        n_classes=4321,
        dim=1234,
        packed_sequence_length=3 * 6 * (512 // 8) ** 2
    )

    images = [
        torch.randn(3, 224, 224),
        torch.randn(3, 32, 512),
        torch.randn(3, 128, 128),
        torch.randn(3, 256, 64),
        torch.randn(3, 64, 256),
        torch.randn(3, 512, 512),
    ]

    out = model(images)
    assert out.shape == (6, 4321), \
        f"{prefix} Head failed. Shape is {out.shape}"

    tokens = model.forward_tokens(images)
    n_tokens = (
        224 * 224 + 32 * 512 + 128 * 128 + 256 * 64 + 64 * 256 + 512 * 512
    ) // 8 ** 2
    assert tokens[0].shape == (1, n_tokens, 1234), \
        f"{prefix} Tokens failed. Shape is {tokens[0].shape}"

    pool = model.forward_pool(*tokens)
    assert pool.shape == (6, 1234), \
        f"{prefix} Pool failed. Shape is {pool.shape}"

    print(f"{prefix} Basic NaViT checks passed.")
