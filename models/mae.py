import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from backbones.vit import (
    ViT,
    Transformer,
    vit_nano,
    vit_micro,
    vit_tiny,
    vit_small,
    vit_base,
    vit_large,
)


class MAE(nn.Module):
    """Masked autoencoder with a vision transformer encoder and decoder

    Parameters
    ----------
    backbone : nn.Module
        Vision transformer backbone
    mask_ratio : float
        Ratio of masked patches
    pre_norm : bool
        Whether to apply layer normalization before the attention layer
    decoder_embed_dim : int
        Dimension of the decoder embedding
    decoder_depth : int
        Number of decoder transformer blocks
    decoder_num_heads : int
        Number of decoder attention heads

    References
    ----------
    1. K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick, "Masked
       Autoencoders Are Scalable Vision Learners". CVPR 2022.
       https://arxiv.org/abs/2111.06377.
    2. https://github.com/lucidrains/vit-pytorch
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_ratio: float = 0.7,
        decoder_dim: int = 768,
        decoder_depth: int = 12,
        decoder_heads: int = 12,
        decoder_dim_head: int = 64,
    ):
        super(MAE, self).__init__()
        self.arguments = locals()

        self.mask_ratio = mask_ratio
        self.decoder_dim = decoder_dim

        self.encoder = backbone

        self.to_patch_embedding = backbone.to_patch_embedding

        self.patch_height = backbone.patch_height
        self.patch_width = backbone.patch_width
        self.n_registers = backbone.n_registers
        self.in_chans = backbone.in_chans

        embed_dim = backbone.dim
        n_patches = backbone.n_patches
        n_patch_pixels = self.patch_height * self.patch_width * self.in_chans

        self.to_decoder = nn.Identity()
        if embed_dim != decoder_dim:
            self.to_decoder = nn.Linear(embed_dim, decoder_dim)

        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4
        )

        self.decoder_pos_embed = nn.Embedding(n_patches, decoder_dim)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim,  n_patch_pixels, bias=True)

    def _patches_to_img(self, x: torch.Tensor) -> torch.Tensor:
        h = self.img_size[0] // self.patch_height
        w = self.img_size[1] // self.patch_width
        ph = self.patch_height
        pw = self.patch_width
        c = self.in_chans

        x = x.view(-1, h, w, ph, pw, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(1, c, h * ph, w * pw)

        return x

    def random_masking(self, tokens: torch.Tensor, mask_ratio: float):
        device = tokens.device
        b, n_patches, *_ = tokens.shape

        n_masked = int((1 - mask_ratio) * n_patches)
        idx = torch.rand(b, n_patches, device=device).argsort(dim=-1)
        mask, unmask = idx[:, :n_masked], idx[:, n_masked:]

        batch_range = torch.arange(b)[:, None]
        tokens = tokens[batch_range, unmask]

        return tokens, mask, unmask

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        b, c, h, w = x.shape
        p1, p2 = self.patch_height, self.patch_width

        ph = h // p1
        pw = w // p2

        x = x.reshape(b, c, ph, p1, pw, p2)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(b, ph * pw, p1 * p2 * c)

        tokens = self.to_patch_embedding(x)

        cls_tokens = self.encoder.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        tokens += self.encoder.pos_embedding[:, :(ph * pw + 1)]
        tokens = self.encoder.dropout(tokens)

        r = self.encoder.register_tokens.expand(b, -1, -1)

        tokens = torch.cat((r, tokens), dim=1)
        tokens = self.encoder.transformer(tokens)
        tokens = tokens[:, self.encoder.n_registers + 1:, :]

        tokens, mask, unmask = self.random_masking(tokens, self.mask_ratio)

        return x, tokens, mask, unmask

    def forward_decoder(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        unmask: torch.Tensor,
    ) -> torch.Tensor:
        device = tokens.device
        b = tokens.shape[0]
        n_masked, n_unmasked = mask.shape[1], unmask.shape[1]

        tokens = self.to_decoder(tokens)
        unmasked_tokens = tokens + self.decoder_pos_embed(unmask)

        mask_tokens = self.mask_token.repeat(b, n_masked, 1)
        mask_tokens = mask_tokens + self.decoder_pos_embed(mask)

        decoder_tokens = torch.zeros(
            b,
            n_masked + n_unmasked,
            self.decoder_dim,
            device=device
        )

        batch_range = torch.arange(b)[:, None]

        decoder_tokens[batch_range, unmask] = unmasked_tokens
        decoder_tokens[batch_range, mask] = mask_tokens

        decoded_tokens = self.decoder(decoder_tokens)
        decoded_tokens = self.decoder_pred(decoded_tokens)

        return decoded_tokens

    def forward_loss(
        self,
        x: torch.Tensor,
        decoded_tokens: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(decoded_tokens, x, reduction='mean')
        return loss

    def forward(
        self,
        x: torch.Tensor,
        with_reconstructed: bool = False
    ) -> torch.Tensor:
        device = x.device
        b, c, h, w = x.shape
        self.img_size = (h, w)

        x, tokens, mask, unmask = self.forward_encoder(x)

        decoded_tokens = self.forward_decoder(tokens, mask, unmask)

        loss = self.forward_loss(x, decoded_tokens)

        if with_reconstructed:
            batch_range = torch.arange(b)[:, None]
            reconstruct = torch.zeros(decoded_tokens.shape, device=device)
            reconstruct[batch_range, unmask] = x[batch_range, unmask]
            reconstruct[batch_range, mask] = decoded_tokens[batch_range, mask]
            return loss, self._patches_to_img(reconstruct)
        else:
            return loss


if __name__ == "__main__":
    nano = vit_nano()
    micro = vit_micro()
    tiny = vit_tiny()
    small = vit_small()
    base = vit_base()
    large = vit_large()

    x = torch.randn(1, 3, 224, 224)
    mae = MAE(nano)
    _, _ = mae(x, with_reconstructed=True)

    mae = MAE(micro)
    _, _ = mae(x, with_reconstructed=True)

    mae = MAE(tiny)
    _, _ = mae(x, with_reconstructed=True)

    mae = MAE(small)
    _, _ = mae(x, with_reconstructed=True)

    mae = MAE(base)
    _, _ = mae(x, with_reconstructed=True)

    mae = MAE(large)
    _, _ = mae(x, with_reconstructed=True)

    print("All tests passed")
