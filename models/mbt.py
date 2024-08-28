import os
import math
import copy
import torch
import datetime
import warnings
import argparse
import torch.nn as nn
import webdataset as wds
import torchvision.transforms.v2 as T

from tqdm import tqdm
from PIL import Image
from typing import Tuple
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from backbones.vit import ViT
from backbones.zoo import select_backbone


def parse_args():
    parser = argparse.ArgumentParser(description="Masked Barlow Twins")

    parser.add_argument(
        "--input", type=str,
        help="Path to input folder or tar file"
    )

    parser.add_argument(
        "--output", type=str,
        help="Path to output folder"
    )

    parser.add_argument(
        "--backbone", type=str, default="vit_small",
        help="Vision transformer backbone"
    )

    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Image height/width"
    )

    parser.add_argument(
        "--channels", type=int, default=3,
        help="Number of image channels"
    )

    parser.add_argument(
        "--patch_size", type=int, default=16,
        help="ViT patch/token height/width"
    )

    parser.add_argument(
        "--mask_ratio_min", type=float, default=0.3,
        help="Minimum ratio of masked patches"
    )

    parser.add_argument(
        "--mask_ratio_max", type=float, default=0.3,
        help="Maximum ratio of masked patches"
    )

    parser.add_argument(
        "--rr_lambda", type=float, default=0.0051,
        help="Coefficient for redundancy reduction term"
    )

    parser.add_argument(
        "--projector_dims", type=int, nargs="+", default=[512, 512, 2048],
        help="Hidden dimensions of the projector head"
    )

    parser.add_argument(
        "--n_views", type=int, default=6,
        help="Number of views per image"
    )

    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size"
    )

    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of workers for data loader"
    )

    parser.add_argument(
        "--n_batches", type=int, default=1000,
        help="Number of batches per epoch (only used if input is tar)"
    )

    parser.add_argument(
        "--epochs", type=int, default=512,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr_min", type=float, default=1e-6,
        help="Minimum learning rate"
    )

    parser.add_argument(
        "--lr_max", type=float, default=0.001,
        help="Maximum learning rate"
    )

    parser.add_argument(
        "--weight_decay", type=float, default=1e-6,
        help="Weight decay for AdamW optimizer"
    )

    parser.add_argument(
        "--lr_warmup", type=float, default=0.1,
        help="Fraction of epochs for cosine scheduler warmup"
    )

    parser.add_argument(
        "--n_checkpoint", type=int, default=None,
        help="Number of epochs between checkpoints"
    )

    parser.add_argument(
        "--print_fraction", type=float, default=0.025,
        help="Fraction of total iterations to print batch loss"
    )

    return parser.parse_args()


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

        loss = (loss_mim + loss_cls) / (2 * b)

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
        crop_scale: Tuple[float, float] = (0.7, 1.0),
        rotation: bool = True,
        hf_p: float = 0.5,
        vf_p: float = 0.5,
        cj_p: float = 0.2,
        cj_b: float = 0.4,
        cj_c: float = 0.4,
        cj_s: float = 0.2,
        cj_h: float = 0.1,
        gs_p: float = 0.2,
        gb_p: float = 0.2,
        gb_k: int = 3,
        gb_s: Tuple[float, float] = (0.1, 1.0),
        gn_p: float = 0.0,
        gn_s: Tuple[float, float] = 0.03,
        iv_p: float = 0.0,
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

        self.augment.extend([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])

        if gn_p > 0.:
            self.augment.append(T.RandomApply([
                T.GaussianNoise(sigma=gn_s)
            ], p=gn_p))

        if iv_p > 0.:
            self.augment.append(T.RandomApply([
                T.Invert()
            ], p=iv_p))

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


class CosineDecay(_LRScheduler):
    """Cosine decay learning rate schedule with linear warmup

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Torch optimizer
    min_lr : float
        Minimum learning rate
    fraction_warmup : float
        Fraction of total iterations to do linear warmup
    iterations : int
        Total number of iterations
    """
    last_iteration: int = -1

    def __init__(
        self,
        optimizer,
        min_lr: float = 0.0001,
        fraction_warmup: int = 0.2,
        iterations: int = 1000,
        last_iteration: int = -1,
    ):
        super(CosineDecay, self).__init__(optimizer, last_iteration)

        assert min_lr >= 0 and isinstance(min_lr, float), \
            f"Expected positive float min_lr, but got {min_lr}"

        assert fraction_warmup >= 0 and fraction_warmup <= 1, \
            f"Expected fraction_warmup in [0, 1], but got {fraction_warmup}"

        self.warmup_steps = int(iterations * fraction_warmup)
        self.cosine_steps = iterations - self.warmup_steps

        self.min_lr = min_lr
        self.iter = 0
        self.total_steps = self.warmup_steps + self.cosine_steps

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )

        if self.iter < self.warmup_steps:
            return [
                base_lr * self.iter / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5
                * (1. + math.cos(
                    math.pi
                    * (self.iter - self.warmup_steps)
                    / (self.total_steps - self.warmup_steps)
                )) for base_lr in self.base_lrs
            ]

    def step(self, iter=None):
        """Step can be called after every batch update or after every epoch"""
        self.iter = self.last_iteration + 1
        self.last_iteration = math.floor(self.iter)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(
                self.optimizer.param_groups,
                self.get_lr()
            )):
                param_group, lr = data
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class FolderDataset(Dataset):
    """Load images from a folder

    Parameters
    ----------
    path : str
        Path to image folder
    transform : object
        Image transformation
    extensions : str
        Valid image file extensions separated by semicolon
    """

    def __init__(
        self,
        path: str,
        transform: object = None,
        extensions: str = "tif;tiff;jpg;jpeg;png;webp",
    ):
        valid_extensions = extensions.split(";")
        paths = os.listdir(path)

        self.image_paths = [
            os.path.join(path, p) for p in paths
            if p.split(".")[-1] in valid_extensions
        ]

        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)


def message(output: str, prefix: str = "INFO", cout: bool = True) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = f"[{prefix} | herb | {timestamp} ] {output}"
    if not cout:
        return out
    print(out, flush=True)


def main(args: argparse.Namespace):
    if args.backbone not in [
        "vit_nano",
        "vit_micro",
        "vit_tiny",
        "vit_small",
        "vit_base",
        "vit_large",
    ]:
        raise ValueError(
            f"Unknown backbone: {args.backbone}. Must be vit_* only."
        )

    if args.output is not None:
        if os.path.isfile(args.output):
            raise ValueError("Output must be a directory")

        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output = os.path.join(args.output, date + "_distill")
        os.makedirs(output, exist_ok=True)

    message(f"Output initialized at {args.output}")

    args.lr_max = args.lr_max * args.batch_size / 256
    args.lr_min = args.lr_min * args.batch_size / 256
    args.batch_size = args.batch_size // args.n_views

    transform = MBTAugmentation(
        args.image_size,
        n_views=args.n_views,
    )

    message(f"Building data loader for {args.input}")

    if os.path.isdir(args.input):
        input_type = "folder"

        dataset = FolderDataset(
            args.input,
            transform=transform,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
        )

        args.n_batches = len(loader)
    elif os.path.isfile(args.input) and ".tar" in args.input:
        input_type = "tar"

        dataset = (
            wds.WebDataset(args.input, shardshuffle=True)
            .shuffle(args.batch_size)
            .decode("pil")
            .to_tuple("tiff;png;jpeg;tif;jpg;webp")
            .map_tuple(transform)
            .batched(args.batch_size)
        )

        loader = wds.WebLoader(dataset, batch_size=None, shuffle=False)
        loader = loader.with_epoch(args.n_batches)
    else:
        raise FileNotFoundError(
            f"Invalid input: {args.input}. Must be a folder or tar file."
        )

    device = "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    message(f"Initializing backbone {args.backbone}")

    backbone = select_backbone(
        args.backbone,
        image_size=args.image_size,
        channels=args.channels,
        patch_size=args.patch_size,
    )

    backbone.head = nn.Identity()

    model = MBT(
        backbone=backbone,
        mask_ratio=(args.mask_ratio_min, args.mask_ratio_max),
        rr_lambda=args.rr_lambda,
        projector_dims=args.projector_dims,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineDecay(
        optimizer,
        min_lr=args.lr_min,
        fraction_warmup=args.lr_warmup,
        iterations=args.epochs * args.n_batches,
    )

    message(f"Starting MBT training on {device}")

    logger = {"train": {"epoch": [], "loss": []}}
    logger["parameters"] = vars(args)

    for epoch in range(args.epochs):
        if epoch > 0:
            print("")

        message(f"Epoch {epoch + 1}")

        model.train()

        progress_bar = tqdm(loader, total=args.n_batches)
        description = message(f"Epoch {epoch + 1}", prefix="LOOP", cout=False)

        running_loss = 0.
        with progress_bar as pb:
            for i, views in enumerate(pb):
                pb.set_description(description)

                if input_type == "tar":
                    views = views[0]

                loss = model(views.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pb.set_postfix({"Loss": loss.item()})

                running_loss += loss.item()

                if device == "cuda":
                    torch.cuda.empty_cache()

                if i % int(args.n_batches * args.print_fraction) == 0:
                    message(f"Batch {i+1} loss: {loss.item()}")

        message(f"Epoch {epoch + 1} loss: {running_loss / args.n_batches}")

        logger["train"]["epoch"].append(epoch)
        logger["train"]["loss"].append(running_loss / args.n_batches)

        _epoch = f"Epoch {epoch+1}"
        message(f"{_epoch} train loss: {running_loss / args.n_batches}")

        del running_loss

        if args.n_checkpoint is not None:
            if (epoch + 1) % args.n_checkpoint == 0:
                check = f"{output}/check_{epoch}"
                torch.save(logger, f"{check}-logger.pt")
                model.encoder.save(f"{check}-encoder.pth")
                model.encoder.save(f"{check}-encoder.safetensors")

    if args.output is not None:
        torch.save(logger, f"{output}/logger.pt")
        model.encoder.save(f"{output}/final_encoder.pth")
        model.encoder.save(f"{output}/final_encoder.safetensors")


if __name__ == "__main__":
    args = parse_args()
    main(args)
