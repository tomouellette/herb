import math
import copy
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T

from PIL import Image
from torch import Tensor
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


class MBT(nn.Module):
    """Masked Barlow Twins

    Parameters
    ----------
    backbone : ViT
        Vision transformer backbone
    mask_ratio : float
        Ratio of masked patches
    rr_lambda : float
        Coefficient for redundancy reduction term
    projector_dims : Tuple[int, ...]
        Hidden dimensions of the projector head

    References
    ----------
    1. J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny. "Barlow Twins:
       Self-Supervised Learning via Redundancy Reduction". ICML 2021.
    2. This repository: this is the first implementation of an iBOT-like
       latent correlation based self-supervised learning method.

    Notes
    -----
    This is an extension of Barlow Twins to include token masking and
    swapped class token prediction in the essence of iBOT.
    """

    def __init__(
        self,
        backbone: ViT,
        mask_ratio: float = (0.1, 0.5),
        rr_lambda: float = 0.0051,
        projector_dims: Tuple[int, ...] = (512, 512, 2048),
    ):
        super(MBT, self).__init__()

        self.mask_ratio = mask_ratio
        self.rr_lambda = rr_lambda

        self.encoder = backbone

        self.to_patch_embedding = backbone.to_patch_embedding

        self.patch_height = backbone.patch_height
        self.patch_width = backbone.patch_width
        self.n_registers = backbone.n_registers
        self.in_chans = backbone.in_chans

        sizes = [backbone.dim, *projector_dims]
        projector = []
        for i in range(len(sizes) - 2):
            projector.extend([
                nn.Linear(sizes[i], sizes[i + 1], bias=False),
                nn.BatchNorm1d(sizes[i + 1]),
                nn.ReLU(inplace=True)
            ])

        projector.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        projector.append(nn.BatchNorm1d(sizes[-1], affine=False))

        self.projector_mim = nn.Sequential(*projector)
        self.projector_cls = copy.deepcopy(self.projector_mim)

    def forward_encoder(self, x: Tensor) -> Tensor:
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

        cls_tokens = tokens[:, self.encoder.n_registers, :]
        tokens = tokens[:, self.encoder.n_registers + 1:, :]

        return tokens, cls_tokens

    def random_masking(self, tokens: Tensor, mask_ratio: float) -> Tensor:
        device = tokens.device
        b, n_patches, *_ = tokens.shape

        n_masked = int((1 - mask_ratio) * n_patches)
        idx = torch.rand(b, n_patches, device=device).argsort(dim=-1)
        unmask = idx[:, n_masked:]

        batch_range = torch.arange(b)[:, None]
        tokens = tokens[batch_range, unmask]

        return tokens

    def forward_masked(self, x: Tensor) -> Tensor:
        tokens, cls_token = self.forward_encoder(x)

        mask_ratio = torch.rand(1) * \
            (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]

        tokens = self.random_masking(tokens, mask_ratio)

        return tokens, cls_token

    def forward_loss(self, c: Tensor) -> Tensor:
        n, m = c.shape
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        off_diag = c.flatten()[:-1].view(n - 1, m + 1)[:, 1:].flatten()
        off_diag = c.pow_(2).sum()

        loss = on_diag + self.rr_lambda * off_diag

        return loss

    def forward(self, views: Tensor) -> Tensor:
        b, n_views, c, h, w = views.shape

        views = views.view(-1, c, h, w)
        t, c = self.forward_masked(views)

        t = t.mean(dim=1)

        t = self.projector_mim(t)
        c = self.projector_cls(c)

        t = t.view(b, n_views, -1).permute(1, 0, 2)
        c = c.view(b, n_views, -1).permute(1, 0, 2)

        nc = math.comb(n_views, 2)

        loss_mim, loss_cls = 0., 0.
        for i in range(n_views):
            for j in range(n_views):
                if i < j:
                    loss_mim += self.forward_loss((t[i].T @ t[j]) / b) / nc
                    loss_cls += self.forward_loss((c[i].T @ c[j]) / b) / nc

        loss = loss_mim + loss_cls

        return loss


class MBTAugmentation:
    """Generic image augmentations

    Parameters
    ----------
    image_size : int
        Image height/width
    crop_scale : Tuple[float, float]
        Min/max scale of random resized crop
    hf_p : float
        Probability of random horizontal flip
    vf_p : float
        Probability of random vertical flip
    cj_p : float
        Probability of color jitter
    cj_b : float
        Brightness jitter factor
    cj_c : float
        Contrast jitter factor
    cj_s : float
        Saturation jitter factor
    cj_h : float
        Hue jitter factor
    gs_p : float
        Probability of random grayscale
    gb_p : float
        Probability of random gaussian blur
    gb_k : int
        Kernel size of gaussian blur
    gb_s : Tuple[float, float]
        Min/max sigma of gaussian blur
    """

    def __init__(
        self,
        image_size: int,
        n_views: int = 4,
        mode: int = "rgb",
        crop_scale: Tuple[float, float] = (0.95, 1.0),
        rotation: bool = True,
        hf_p: float = 0.5,
        vf_p: float = 0.5,
        cj_p: float = 0.5,
        cj_b: float = 0.4,
        cj_c: float = 0.4,
        cj_s: float = 0.2,
        cj_h: float = 0.1,
        gs_p: float = 0.2,
        gb_p: float = 0.2,
        gb_k: int = 3,
        gb_s: Tuple[float, float] = (0.1, 2.0),
    ):
        self.n_views = n_views

        self.augment = [
            T.RandomResizedCrop(
                image_size,
                scale=crop_scale,
                interpolation=Image.BICUBIC)
        ]

        if rotation:
            self.augment.append(T.RandomChoice([
                T.RandomRotation(i, interpolation=Image.BICUBIC)
                for i in range(0, 360, 90)
            ]))

        if hf_p > 0.:
            self.augment.append(T.RandomHorizontalFlip(p=hf_p))

        if vf_p > 0.:
            self.augment.append(T.RandomVerticalFlip(p=vf_p))

        if cj_p > 0.:
            self.augment.append(T.RandomApply([
                T.ColorJitter(
                    brightness=cj_b,
                    contrast=cj_c,
                    saturation=cj_s,
                    hue=cj_h,
                )], p=cj_p,
            ))

        if gs_p > 0.:
            self.augment.append(T.RandomGrayscale(p=gs_p))

        if gb_p > 0.:
            self.augment.append(T.RandomApply([
                T.GaussianBlur(gb_k, gb_s)
            ], p=gb_p))

        self.augment.append(T.ToImage())

        if mode == "rgb":
            self.augment.append(T.RGB())
            normalize = T.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.25, 0.25, 0.25),
            )
        elif mode == "gray":
            self.augment.append(T.Grayscale())
            normalize = T.Normalize(
                mean=(0.5,),
                std=(0.25,),
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.augment.extend([
            T.ToDtype(torch.float32, scale=True),
            normalize,
        ])

        self.augment = T.Compose(self.augment)

    def __call__(self, x):
        views = [self.augment(x) for _ in range(self.n_views)]
        return torch.stack(views)
