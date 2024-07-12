import copy
import math
import torch
import argparse
import warnings
import datetime
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch import Tensor
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.optim.lr_scheduler import _LRScheduler
from typing import Tuple, List

from accelerate import Accelerator
from accelerate.utils import tqdm

from backbones.zoo import select_backbone


# -----------------------------------------------------------------------------
#                                   Comments
# -----------------------------------------------------------------------------


# This is a custom DINO implementation that differs a tad from original paper:
#
# - Augmentation:
#   - Original paper does 2 global crops and N local crops on input image
#     and exposes teacher to only global crops and student to local crops
#     under the pretense that learning local-to-global correspondence is
#     a good pretext task for vision
#   - This implementation simply does N even random augmentations on an
#     an input image and then just splits the augments evenly across the
#     teacher and student networks; this actually simplifies the forward
#     pass since views can just be resized and chunked into tensor input;
#     plus it's not entirely clear how much augmentation is actually needed
#     for performant self-supervised pre-training with joint embedding models
#     (see https://arxiv.org/abs/2406.09294 for example)
#
# - Scheduling:
#   - Original paper performs cosine scheduling on the learning rate
#     and weight decay of the optimizer during training, and cosine
#     scheduling on the momentum of the teacher network
#   - This implementation only does cosine scheduling on the learning rate
#     and fixes the weight decay to a constant value; the momentum on the
#     teacher network is annealed to 1 using a linear schedule
#
# - Training:
#   - Original paper clips gradients and freezes the projector on the N epochs
#     and then unfreezes it on N + 1 epoch
#   - This implementation doesn't freeze the projector on the first epoch and
#     doesn't clip gradients in the student network during training; I haven't
#     found too much instability with this implementation but if I do I'll add
#     gradient clipping and projector freezing
#
#
# Papers/resources that have been useful for building this DINO implementation:
#
# - "Emerging Properties in Self-Supervised Vision Transformers"
#   Caron et al. (2021). https://arxiv.org/abs/2104.14294
# - "You Don't Need Data-Augmentation in Self-Supervised Learning"
#   Moutkanni et al. (2024). https://arxiv.org/abs/2406.09294
# - The original DINO implementation:
#   https://github.com/facebookresearch/dino/
# - Lucidrains DINO implementation:
#   https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/dino.py
#
# Test usage:
#
# - From herb root: python3 -m models.dino --test True --epochs 1
#

# -----------------------------------------------------------------------------
#                                   Modules
# -----------------------------------------------------------------------------


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


class Augmentation(object):
    """A default augmentation pipeline for DINO

    Parameters
    ----------
    image_size : int
        Image size of data
    global_crops_scale : tuple[float, float]
        Scale for global crops
    local_crops_scale : tuple[float, float]
        Scale for local crops
    n_augments : int
        Even number of crops
    """

    def __init__(
        self,
        image_size: int,
        channels: int,
        global_crops_scale: tuple,
        local_crops_scale: tuple,
        n_augments: int
    ):
        if n_augments % 2 != 0:
            raise ValueError("n_augments must be an even number")

        self.n_augments = n_augments // 2

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
            T.ToTensor(),
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

    def __call__(self, image: Image) -> List[Tensor]:
        crops = []
        for _ in range(self.n_augments):
            crops.append(self.global_crops(image))
            crops.append(self.local_crops(image))

        return torch.stack(crops)


class StudentTeacher(nn.Module):
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
        momentum_center: float = 0.9
    ):
        super().__init__()
        x = torch.randn(1, *image_size)
        d = backbone(x).flatten(1).shape[-1]

        self.student_encoder = backbone
        self.student_projector = Projector(
            input_dim=d,
            projector_k=projector_k,
            projector_hidden_dim=projector_hidden_dim,
            projector_layers=projector_layers,
            use_bn=projector_batch_norm,
            l2_norm=projector_l2_norm
        )

        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_projector = Projector(
            input_dim=d,
            projector_k=projector_k,
            projector_hidden_dim=projector_hidden_dim,
            projector_layers=projector_layers,
            use_bn=projector_batch_norm,
            l2_norm=False
        )

        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)

        for p in self.teacher_projector.parameters():
            p.requires_grad_(False)

        self.momentum_center = momentum_center
        self.register_buffer("center", torch.zeros(1, projector_k))

    @torch.no_grad()
    def update_ema(self, momentum: float):
        for pt, ps in zip(
            self.teacher_encoder.parameters(),
            self.student_encoder.parameters()
        ):
            pt.data = pt.data * momentum + ps.data * (1 - momentum)

        for pt, ps in zip(
            self.teacher_projector.parameters(),
            self.student_projector.parameters()
        ):
            pt.data = pt.data * momentum + ps.data * (1 - momentum)

    @torch.no_grad()
    def update_centers(self, teacher_output: Tensor):
        self.center = self.center.to(teacher_output.device)
        self.center = self.center * self.momentum_center + \
            teacher_output.mean(dim=0) * (1 - self.momentum_center)
        return self.center

    def forward(self, images: Tensor) -> Tensor:
        batch, n_augments, c, h, w = images.shape
        n_pairs = n_augments // 2

        vs = images[:, :n_pairs]
        vs = vs.reshape(batch * n_pairs, c, h, w)
        vs = self.student_encoder(vs)
        vs = self.student_projector(vs)

        with torch.no_grad():
            vt = images[:, n_pairs:]
            vt = vt.reshape(batch * n_pairs, c, h, w)
            vt = self.teacher_encoder(vt)
            vt = self.teacher_projector(vt)

        centers = self.update_centers(vt)

        return vs, vt, centers


class StudentTeacherLoss(nn.Module):
    """A loss function for student-teacher training

    Parameters
    ----------
    epochs : int
        Number of epochs
    projector_k : int
        Output dimension of the projector (i.e. prototype count)
    t_teacher_start : float
        Initial temperature of teacher network
    t_teacher_end : float
        Final temperature of teacher network
    t_teacher_warmup_fraction : int
        Fraction of total iterations to do linear warmup
    t_student : float
        Temperature of student network
    """

    def __init__(
        self,
        epochs: int,
        projector_k: int,
        t_teacher_start: float = 0.04,
        t_teacher_end: float = 0.02,
        t_teacher_warmup_fraction: int = 0.1,
        t_student: float = 0.1,
    ):
        super().__init__()
        t_teacher_warmup = int(epochs * t_teacher_warmup_fraction)

        self.anneal_t_teacher = [
            t_teacher_start
            + (t_teacher_end - t_teacher_start)
            * i / t_teacher_warmup
            for i in range(t_teacher_warmup)
        ] + [t_teacher_end] * (epochs - t_teacher_warmup + 1)

        self.t_student = t_student

    def forward(
        self,
        student_output: Tensor,
        teacher_output: Tensor,
        centers: Tensor,
        epoch: int,
        eps: float = 1e-20
    ) -> Tensor:
        p_teacher = teacher_output.detach()
        p_teacher = p_teacher - centers
        p_teacher /= self.anneal_t_teacher[epoch]
        p_teacher = p_teacher.softmax(dim=-1)

        p_student = student_output / self.t_student
        p_student = F.log_softmax(p_student + eps, dim=-1)

        loss = - (p_teacher * p_student).sum(dim=-1).mean()

        return loss


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

# -----------------------------------------------------------------------------
#                                   Functions
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
#                                   Testing
# -----------------------------------------------------------------------------


def test_mnist(batch_size: int = 64, transform=None):
    """Load MNIST data"""
    import os
    import ssl
    import shutil
    import numpy as np
    from torchvision.datasets import MNIST
    from sklearn.manifold import TSNE
    from torch.utils.data import DataLoader
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    if os.path.exists("tests/dino/"):
        shutil.rmtree("tests/dino/")

    os.makedirs("tests/dino/")

    ssl._create_default_https_context = ssl._create_unverified_context

    train = MNIST(
        root="data", train=True, download=True, transform=transform
    )

    test = MNIST(
        root="data", train=False, download=True, transform=test_transform
    )

    train_loader, test_loader = (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )

    @torch.no_grad()
    def evaluation_function(
        student_teacher,
        test_loader,
        epoch,
        device,
        finalize: bool = False,
    ):
        GIF = type(1)(os.getenv('GIF', 1))

        if GIF and finalize:
            import imageio

        student_teacher.eval()
        tz, sz, y = [], [], []
        with tqdm(test_loader) as progress:
            description = message(f"Epoch {epoch} | Testing", cout=False)
            description = description.replace("INFO", "TEST")
            for xi, yi in progress:
                progress.set_description(description)
                tzi = student_teacher.teacher_encoder(xi.to(device))
                szi = student_teacher.student_encoder(xi.to(device))
                tzi = tzi.flatten(1)
                szi = szi.flatten(1)
                tz.append(tzi.detach().cpu().numpy())
                sz.append(szi.detach().cpu().numpy())
                y.append(yi.numpy())

        tz = np.concatenate(tz, axis=0)
        sz = np.concatenate(sz, axis=0)
        y = np.concatenate(y, axis=0)

        tz = (tz - tz.mean(axis=0)) / tz.std(axis=0)
        sz = (sz - sz.mean(axis=0)) / sz.std(axis=0)

        fig, ax = plt.subplots(1, 2, figsize=(5, 2))
        tpca = PCA(n_components=2).fit_transform(tz)
        spca = PCA(n_components=2).fit_transform(sz)
        ax[0].scatter(*tpca.T, c=y, cmap="tab10", alpha=0.5, s=1.5)
        ax[1].scatter(*spca.T, c=y, cmap="tab10", alpha=0.5, s=1.5)
        ax[0].set_title("Teacher PCA")
        ax[1].set_title("Student PCA")
        ax[0].axis("off")
        ax[1].axis("off")
        fig.tight_layout()
        plt.savefig(f"tests/dino/pca_{epoch}.png")
        plt.close()

        np.random.seed(123456)
        for z, name in zip([tz, sz], ["teacher", "student"]):
            logreg = LogisticRegression(max_iter=1000)
            idx = np.random.choice(
                z.shape[0],
                size=int(0.5 * z.shape[0]),
                replace=False
            )

            x_train = z[idx]
            y_train = y[idx]
            x_test = z[~idx]
            y_test = y[~idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logreg.fit(x_train, y_train)

            y_pred = logreg.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            message(f"Epoch {epoch} {name} | Accuracy: {acc}")

        if finalize:
            if GIF:
                images = []
                plot_dir = "tests/dino/"
                for i in range(0, epoch):
                    filename = os.path.join(plot_dir, f'pca_{i}.png')
                    images.append(imageio.v3.imread(filename))

                for _ in range(20):
                    images.append(imageio.v3.imread(filename))

                imageio.v3.imwrite(
                    os.path.join(plot_dir, 'training.gif'),
                    images,
                    loop=1000
                )

            ttsne = TSNE(n_components=2, n_jobs=-1).fit_transform(tz)
            stsne = TSNE(n_components=2, n_jobs=-1).fit_transform(sz)

            fig, ax = plt.subplots(1, 2, figsize=(5, 2))
            ax[0].scatter(*ttsne.T, c=y, cmap="tab10", alpha=0.5, s=1.5)
            ax[1].scatter(*stsne.T, c=y, cmap="tab10", alpha=0.5, s=1.5)
            ax[0].set_title("Teacher")
            ax[1].set_title("Student")
            ax[0].axis("off")
            ax[1].axis("off")
            fig.tight_layout()
            plt.savefig(f"tests/dino/tsne_{epoch}.png")
            plt.close()

    return train_loader, test_loader, evaluation_function


# -----------------------------------------------------------------------------
#                                   Training
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        "Self-supervised learning with distillation with no labels"
    )

    parser.add_argument(
        "--image_folder", type=str,
        help="Path to image folder"
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
        "--backbone", type=str, default="mlp_mixer_small",
        help="Backbone model"
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
        "--max_lr", type=float, default=1e-4,
        help="Maximum learning rate"
    )

    parser.add_argument(
        "--min_lr", type=float, default=1e-6,
        help="Minimum learning rate"
    )

    parser.add_argument(
        "--lr_warmup_fraction", type=float, default=0.1,
        help="Fraction of total iterations to do linear warmup"
    )

    parser.add_argument(
        '--weight_decay', type=float, default=0.05,
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
        "--n_augments", type=int, default=4,
        help="Number of augmentations per image (must be even number)"
    )

    parser.add_argument(
        "--t_teacher_start", type=float, default=0.04,
        help="Initial temperature of teacher network"
    )

    parser.add_argument(
        "--t_teacher_end", type=float, default=0.02,
        help="Final temperature of teacher network"
    )

    parser.add_argument(
        "--t_teacher_warmup_fraction", type=float, default=0.1,
        help="Fraction of total iterations for teacher temp. linear warmup"
    )

    parser.add_argument(
        "--t_student", type=float, default=0.1,
        help="Temperature of student network"
    )

    parser.add_argument(
        "--silent", type=bool, default=False,
        help="Disable the progress bar"
    )

    parser.add_argument(
        "--test", type=bool, default=False,
        help="Run test on MNIST"
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    if args.test:
        args.image_size = 28
        args.channels = 1

    transform = Augmentation(
        args.image_size,
        args.channels,
        args.global_crops_scale,
        args.local_crops_scale,
        args.n_augments
    )

    if not args.test:
        backbone = select_backbone(
            model=args.backbone,
            image_size=args.image_size,
            channels=args.channels
        )

        dataset = datasets.ImageFolder(
            args.image_folder,
            transform=transform
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
    else:
        backbone = select_backbone(
            model="mlp_mixer_tiny",
            image_size=args.image_size,
            channels=args.channels,
            patch_size=7
        )

        loader, test_loader, evaluation_function = test_mnist(
            args.batch_size,
            transform=transform
        )

    backbone.head = nn.Identity()

    model = StudentTeacher(
        backbone,
        (args.channels, args.image_size, args.image_size),
        args.projector_hidden_dim,
        args.projector_k,
        args.projector_layers,
        args.projector_batch_norm,
        args.projector_l2_norm,
        args.momentum_center
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineDecay(
        optimizer,
        args.min_lr,
        args.lr_warmup_fraction,
        iterations=len(loader) * args.epochs,
    )

    ema_schedule = momentum_schedule(
        args.momentum_teacher,
        0.9999,
        len(loader) * args.epochs
    )

    loss_function = StudentTeacherLoss(
        args.epochs,
        args.projector_k,
        args.t_teacher_start,
        args.t_teacher_end,
        args.t_teacher_warmup_fraction,
        args.t_student,
    )

    accelerator = Accelerator()
    dataloader, model, optimizer, scheduler = accelerator.prepare(
        loader, model, optimizer, scheduler
    )

    device = accelerator.device

    message(f"Training started on device: {device}")

    for epoch in range(args.epochs):
        if epoch > 0:
            print("")

        message(f"Epoch {epoch+1}")

        progress_bar = tqdm(
            loader,
            disable=not accelerator.is_local_main_process and args.silent
        )

        description = message(f"Epoch {epoch+1}", cout=False)
        description = description.replace("INFO", "LOOP")

        with progress_bar as progress:
            for images, *_ in progress:
                progress.set_description(description)

                images = images.to(device)
                student_output, teacher_output, centers = model(images)

                loss = loss_function(
                    student_output,
                    teacher_output,
                    centers,
                    epoch
                )

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                model.update_ema(next(ema_schedule))

                progress.set_postfix({"Loss": loss.item()})

        if args.test:
            evaluation_function(
                model,
                test_loader,
                epoch,
                device,
                finalize=epoch == args.epochs - 1
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
