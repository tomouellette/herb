import os
import math
import torch
import datetime
import warnings
import argparse
import torch.nn as nn
import webdataset as wds
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from PIL import Image
from typing import Tuple
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from backbones.vit import ViT, Transformer
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
        "--mask_ratio", type=float, default=0.7,
        help="Ratio of masked patches"
    )

    parser.add_argument(
        "--decoder_dim", type=int, default=None,
        help="Dimension of the decoder embedding"
    )

    parser.add_argument(
        "--decoder_depth", type=int, default=None,
        help="Number of decoder transformer blocks"
    )

    parser.add_argument(
        "--decoder_heads", type=int, default=None,
        help="Number of decoder attention heads"
    )

    parser.add_argument(
        "--decoder_dim_head", type=int, default=None,
        help="Dimension of decoder attention head"
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
        backbone: ViT,
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

    def _patches_to_img(self, x: Tensor) -> Tensor:
        h = self.img_size[0] // self.patch_height
        w = self.img_size[1] // self.patch_width
        ph = self.patch_height
        pw = self.patch_width
        c = self.in_chans

        x = x.view(-1, h, w, ph, pw, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(1, c, h * ph, w * pw)

        return x

    def random_masking(self, tokens: Tensor, mask_ratio: float):
        device = tokens.device
        b, n_patches, *_ = tokens.shape

        n_masked = int((1 - mask_ratio) * n_patches)
        idx = torch.rand(b, n_patches, device=device).argsort(dim=-1)
        mask, unmask = idx[:, :n_masked], idx[:, n_masked:]

        batch_range = torch.arange(b)[:, None]
        tokens = tokens[batch_range, unmask]

        return tokens, mask, unmask

    def forward_encoder(self, x: Tensor) -> Tuple[Tensor, ...]:
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
        tokens: Tensor,
        mask: Tensor,
        unmask: Tensor,
    ) -> Tensor:
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
        x: Tensor,
        decoded_tokens: Tensor,
    ) -> Tensor:
        loss = F.mse_loss(decoded_tokens, x, reduction='mean')
        return loss

    def forward(
        self,
        x: Tensor,
        with_reconstructed: bool = False
    ) -> Tensor:
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


class MAEAugmentation:
    """Generic image augmentations

    Parameters
    ----------
    image_size : int
        Image height/width
    crop_scale : Tuple[float, float]
        Min/max scale of random resized crop
    mode : str
        Image mode (rgb or gray)
    rotation : bool
        Random rotation
    """

    def __init__(
        self,
        image_size: int,
        mode: int = "rgb",
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        rotation: bool = True,
    ):
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

        self.augment.extend([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])

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
        return self.augment(x)


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

    transform = MAEAugmentation(args.image_size)

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

    if args.decoder_dim is None:
        args.decoder_dim = backbone.dim

    if args.decoder_depth is None:
        args.decoder_depth = backbone.depth

    if args.decoder_heads is None:
        args.decoder_heads = backbone.heads

    if args.decoder_dim_head is None:
        args.decoder_dim_head = backbone.dim_head

    model = MAE(
        backbone=backbone,
        mask_ratio=args.mask_ratio,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        decoder_dim_head=args.decoder_dim_head,
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

        running_loss = 0.
        for i, views in enumerate(loader):
            if input_type == "tar":
                views = views[0]

            loss = model(views.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if device == "cuda":
                torch.cuda.empty_cache()

            if args.print_fraction > 0:
                if i % int(args.n_batches * args.print_fraction) == 0:
                    message(f"Batch {i+1} loss: {loss.item()}", prefix="LOOP")

            if i >= args.n_batches:
                break

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
