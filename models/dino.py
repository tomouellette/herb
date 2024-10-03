import os
import math
import copy
import torch
import datetime
import warnings
import argparse
import torch.nn as nn
import webdataset as wds
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from PIL import Image
from torch import Tensor
from typing import Tuple, List
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from backbones.zoo import select_backbone


def parse_args():
    parser = argparse.ArgumentParser("Distillation with no labels")

    parser.add_argument(
        "--input", type=str,
        help="Path to image folder"
    )

    parser.add_argument(
        "--output", type=str,
        help="Path to output folder"
    )

    parser.add_argument(
        "--image_size", type=int, default=64,
        help="Image size"
    )

    parser.add_argument(
        "--channels", type=int, default=3,
        help="Number of channels"
    )

    parser.add_argument(
        "--backbone", type=str, default="vit_small",
        help="Backbone model"
    )

    parser.add_argument(
        "--patch_size", type=int, default=16,
        help="Patch size if backbone is vision transformer"
    )

    parser.add_argument(
        "--projector_hidden_dim", type=int, default=256,
        help="Hidden dimension of the projector"
    )

    parser.add_argument(
        "--projector_k", type=int, default=256,
        help="Output dimension of the projectors (i.e. prototype count)"
    )

    parser.add_argument(
        "--projector_layers", type=int, default=4,
        help="Number of layers in the projectors"
    )

    parser.add_argument(
        "--projector_batch_norm", type=bool, default=False,
        help="Use batch normalization in the projectors"
    )

    parser.add_argument(
        "--projector_l2_norm", type=bool, default=False,
        help="Use L2 normalization in the projectors"
    )

    parser.add_argument(
        "--momentum_center", type=float, default=0.9,
        help="Momentum for center EMA update"
    )

    parser.add_argument(
        "--momentum_teacher", type=float, default=0.996,
        help="Momentum for teacher EMA update"
    )

    parser.add_argument(
        "--global_crops_scale", type=float, default=(0.5, 1.0),
        help="Global crop scales",
        nargs="+"
    )

    parser.add_argument(
        "--local_crops_scale", type=float, default=(0.3, 0.7),
        help="Local crop scales",
        nargs="+"
    )

    parser.add_argument(
        "--n_views", type=int, default=4,
        help="Number of augmentations per image (must be even number)"
    )

    parser.add_argument(
        "--t_student", type=float, default=0.1,
        help="Temperature of student network"
    )

    parser.add_argument(
        "--t_teacher", type=float, default=0.04,
        help="Temperature of teacher network"
    )

    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Number of epochs"
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
        "--lr_max", type=float, default=1e-4,
        help="Maximum learning rate"
    )

    parser.add_argument(
        "--lr_min", type=float, default=1e-6,
        help="Minimum learning rate"
    )

    parser.add_argument(
        "--lr_warmup", type=float, default=0.1,
        help="Fraction of total iterations to do linear warmup"
    )

    parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help="Weight decay for AdamW optimizer"
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


class Projector(nn.Module):
    """A MLP project for student and teacher networks.

    Parameters
    ----------
    input_dim : int
        Flattened input dimension
    projector_k : int
        Output dimension
    projector_hidden_dim : int
        Hidden dimension
    projector_layers : int
        Number of layers in the projector
    use_bn : bool
        Whether to use batch normalization
    l2_norm : bool
        Whether to l2-normalize the final output
    """

    def __init__(
        self,
        input_dim: int,
        projector_k: int,
        projector_hidden_dim: int = 256,
        projector_layers: int = 3,
        use_bn: bool = False,
        l2_norm: bool = True
    ):
        super().__init__()
        self.l2_norm = l2_norm

        layers = []
        for i in range(projector_layers-1):
            layers.append(nn.Linear(
                input_dim if i == 0 else projector_hidden_dim,
                projector_hidden_dim
            ))

            if use_bn:
                layers.append(nn.BatchNorm1d(projector_hidden_dim))

            if i != projector_layers - 2:
                layers.append(nn.GELU())

        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(projector_hidden_dim, projector_k)

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x.flatten(1))
        if self.l2_norm:
            norm = x.norm(dim=1, keepdim=True).clamp(min=1e-6)
            x = x / norm

        return self.head(x)


class DINO(nn.Module):
    """A container for passing backbone embedding to linear projector

    Parameters
    ----------
    backbone : nn.Module
        Backbone network (e.g. ViT, ResNet, etc)
    image_size : Tuple[int, int, int]
        Image size of data (c, h, w)
    projector_hidden_dim : int
        Hidden dimension of the projector
    projector_k : int
        Output dimension of the projector (i.e. prototype count)
    projector_layers : int
        Number of layers in the projector
    projector_batch_norm : bool
        Whether to use batch normalization in the projector
    projector_l2_norm : bool
        Whether to l2-normalize the final output
    momentum_center : float
        Momentum for center EMA update
    """

    def __init__(
        self,
        backbone: nn.Module,
        image_size: Tuple[int, int, int],
        projector_hidden_dim: int = 256,
        projector_k: int = 256,
        projector_layers: int = 4,
        projector_batch_norm: bool = False,
        projector_l2_norm: bool = False,
        n_views: int = 5,
        t_student: float = 0.1,
        t_teacher: float = 0.04,
        momentum_center: float = 0.9,
    ):
        super().__init__()
        x = torch.randn(1, *image_size)
        d = backbone(x).flatten(1).shape[-1]

        self.se = backbone
        self.sp = Projector(
            input_dim=d,
            projector_k=projector_k,
            projector_hidden_dim=projector_hidden_dim,
            projector_layers=projector_layers,
            use_bn=projector_batch_norm,
            l2_norm=projector_l2_norm
        )

        self.te = copy.deepcopy(self.se)
        self.tp = Projector(
            input_dim=d,
            projector_k=projector_k,
            projector_hidden_dim=projector_hidden_dim,
            projector_layers=projector_layers,
            use_bn=projector_batch_norm,
            l2_norm=False
        )

        for p in self.te.parameters():
            p.requires_grad_(False)

        for p in self.tp.parameters():
            p.requires_grad_(False)

        self.n_views = n_views
        self.t_student = t_student
        self.t_teacher = t_teacher

        self.momentum_center = momentum_center
        self.register_buffer("center", torch.zeros(1, projector_k))

    @torch.no_grad()
    def update_centers(self, teacher_output: Tensor):
        self.center = self.center.to(teacher_output.device)
        self.center = self.center * self.momentum_center + \
            teacher_output.mean(dim=0) * (1 - self.momentum_center)
        return self.center

    @torch.no_grad()
    def update_ema(self, momentum: float):
        for t, s in zip(
            self.tp.parameters(),
            self.sp.parameters()
        ):
            t.data = t.data * momentum + s.data * (1 - momentum)

    def forward(self, batch: Tensor, device: str) -> Tensor:
        with torch.no_grad():
            vt = [self.tp(self.te(i[0].to(device))) for i in batch]
            vt = torch.vstack(vt).detach()

        vs = [self.sp(self.se(i[1].to(device))) for i in batch]
        vs = torch.vstack(vs)

        s_chunk = vs.shape[0] // (self.n_views + 1)
        t_chunk = vt.shape[0] // 2

        pt = vt - self.center
        pt /= self.t_teacher
        pt = pt.softmax(dim=-1)

        ps = vs / self.t_student
        ps = F.log_softmax(ps + 1e-20, dim=-1)

        pt = pt.chunk(t_chunk)
        ps = ps.chunk(s_chunk)

        loss, count = 0.0, 0
        for s, t in zip(ps, pt):
            for i in range(2):
                for j in range(self.n_views):
                    loss += (-t[i] * s[j]).sum(dim=-1).mean()
                    count += 1

        loss /= count

        self.center = self.update_centers(vt)

        return loss


class DINOAugmentation(object):
    """A default augmentation pipeline for DINO

    Parameters
    ----------
    image_size : int
        Image size of data
    global_crops_scale : tuple[float, float]
        Scale for global crops
    local_crops_scale : tuple[float, float]
        Scale for local crops
    n_views : int
        Number of local crops
    """

    def __init__(
        self,
        image_size: int,
        channels: int,
        global_crops_scale: tuple = (0.5, 1.0),
        local_crops_scale: tuple = (0.1, 0.8),
        n_views: int = 6
    ):
        if n_views % 2 != 0:
            raise ValueError("n_views must be an even number")

        self.n_views = n_views

        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2, hue=0.1
                )],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        _normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) \
            if channels == 3 else ((0.5,), (0.5,))

        normalize = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(*_normalize)
        ])

        self.global_crops = T.Compose([
            T.RandomResizedCrop(
                image_size,
                scale=global_crops_scale,
                interpolation=Image.BICUBIC
            ),
            flip_and_color_jitter,
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            normalize,
        ])

        self.local_crops = T.Compose([
            T.RandomResizedCrop(
                image_size,
                scale=local_crops_scale,
                interpolation=Image.BICUBIC
            ),
            flip_and_color_jitter,
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            normalize,
        ])

    def __call__(self, image: Image) -> Tensor:
        global_crops = []
        for _ in range(2):
            global_crops.append(self.global_crops(image))

        crops = []
        for _ in range(self.n_views):
            crops.append(self.local_crops(image))

        return torch.stack(global_crops), torch.stack(crops)


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


def message(output: str, cout: bool = True) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = f"[INFO | herb | {timestamp} ] {output}"
    if not cout:
        return out
    print(out)


def momentum_schedule(
    momentum_start: float,
    momentum_end: float,
    iterations: int
) -> List[float]:
    return (
        momentum_start
        + (momentum_end - momentum_start)
        * i / iterations
        for i in range(iterations)
    )


def main(args: argparse.Namespace):
    if args.output is not None:
        if os.path.isfile(args.output):
            raise ValueError("Output must be a directory")

        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output = os.path.join(args.output, date + "_distill")
        os.makedirs(output, exist_ok=True)

    message(f"Output initialized at {args.output}")

    args.lr_max = args.lr_max * args.batch_size / 256
    args.lr_min = args.lr_min * args.batch_size / 256
    args.batch_size = max(args.batch_size // args.n_views, 1)

    transform = DINOAugmentation(
        args.image_size,
        args.channels,
        args.global_crops_scale,
        args.local_crops_scale,
        args.n_views
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

    kwargs = {}
    if "vit" in args.backbone:
        kwargs["patch_size"] = args.patch_size

    backbone = select_backbone(
        args.backbone,
        image_size=args.image_size,
        channels=args.channels,
        **kwargs
    )

    backbone.head = nn.Identity()

    model = DINO(
        backbone,
        (args.channels, args.image_size, args.image_size),
        args.projector_hidden_dim,
        args.projector_k,
        args.projector_layers,
        args.projector_batch_norm,
        args.projector_l2_norm,
        args.n_views,
        args.t_student,
        args.t_teacher,
        args.momentum_center,
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

    ema_schedule = momentum_schedule(
        args.momentum_teacher,
        0.9999,
        args.n_batches * args.epochs
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

            loss = model(views, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            model.update_ema(next(ema_schedule))

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
                model.se.save(f"{check}-student.pth")
                model.te.save(f"{check}-teacher.pth")
                model.se.save(f"{check}-student.safetensors")
                model.te.save(f"{check}-teacher.safetensors")

    if args.output is not None:
        torch.save(logger, f"{output}/logger.pt")
        model.se.save(f"{output}/final_student.pth")
        model.te.save(f"{output}/final_teacher.pth")
        model.se.save(f"{output}/final_student.safetensors")
        model.te.save(f"{output}/final_teacher.safetensors")


if __name__ == "__main__":
    args = parse_args()
    main(args)
