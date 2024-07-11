import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch import Tensor
from torchvision import transforms as T
from typing import Optional, Union, Tuple, List


from accelerate import Accelerator

# -----------------------------------------------------------------------------
#                                   Comments
# -----------------------------------------------------------------------------

# This is a general DINO implementation to get started
# without worrying about image resizing and the additional
# complications of computing loss at different scales. This
# implementation differs from the original in that N even
# augmentations are applied to an input image and then split
# evenly into pairs for the student and teacher networks. The
# original paper does 2 global crops and N local crops where
# the teacher is never exposed to local information; I don't
# do that here. Doing it in resized pairs actually makes it easy to
# just chunk input and reshape input tensors for the student and
# teacher networks. Plus, there's growing evidence that crazy amounts
# of augmentations aren't generally necessary for getting performant
# models from joint embedding methods like DINO (see for example the
# paper https://arxiv.org/abs/2406.09294). This DINO setup should be
# self-contained with all the functionality that is needed inside this
# file; except for a backbone network

# model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
# accelerator.prepare(model)

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
    n_crops : int
        Even number of crops
    """

    def __init__(
        self,
        image_size: int,
        global_crops_scale: tuple,
        local_crops_scale: tuple,
        n_crops: int
    ):
        if n_crops % 2 != 0:
            raise ValueError("n_crops must be an even number")

        self.n_crops = n_crops // 2

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

        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
        for _ in range(self.n_crops):
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
        self.student_projector = Projector(
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

    def forward(self, images: Tensor) -> Tensor:
        batch, n_crops, c, h, w = images.shape
        n_pairs = n_crops // 2

        vs = images[:, :n_pairs]
        vs = vs.reshape(batch * n_pairs, c, h, w)
        vs = self.student_encoder(images[:n_pairs])
        vs = self.student_projector(vs)

        with torch.no_grad():
            vt = images[:, n_pairs:]
            vt = vt.reshape(batch * n_pairs, c, h, w)
            vt = self.teacher_encoder(images[n_pairs:])
            vt = self.teacher_projector(vt)

        return vs, vt


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
    t_teacher_warmup : int
        Number of epochs to warmup teacher temperature
    t_student : float
        Temperature of student network
    momentum_center : float
        Momentum for center EMA update
    """

    def __init__(
        self,
        epochs: int,
        projector_k: int,
        t_teacher_start: float = 0.04,
        t_teacher_end: float = 0.02,
        t_teacher_warmup: int = 10,
        t_student: float = 0.1,
        momentum_center: float = 0.9
    ):
        super().__init__()
        self.anneal_t_teacher = [
            t_teacher_start
            + (t_teacher_end - t_teacher_start)
            * i / t_teacher_warmup
            for i in range(t_teacher_warmup)
        ] + [t_teacher_end] * (epochs - t_teacher_warmup + 1)

        self.t_student = t_student

        self.momentum_center = momentum_center
        self.register_buffer("center", torch.zeros(1, projector_k))

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor):
        self.center = self.center * self.momentum_center + \
            teacher_output.mean(dim=0) * (1 - self.momentum_center)

    def forward(
        self,
        student_output: Tensor,
        teacher_output: Tensor,
        epoch: int,
        eps: float = 1e-20
    ) -> Tensor:
        p_teacher = teacher_output.detach()
        p_teacher = p_teacher - self.center
        p_teacher /= self.anneal_t_teacher[epoch]
        p_teacher = p_teacher.softmax(dim=-1)

        p_student = student_output / self.t_student
        p_student = F.log_softmax(p_student + eps, dim=-1)

        loss = - (p_teacher * p_student).sum(dim=-1).mean()

        self.update_center(teacher_output)

        return loss


# -----------------------------------------------------------------------------
#                                   Functions
# -----------------------------------------------------------------------------

The verbose parameter is deprecated. Please use get_last_lr()
def cosine_scheduler(
    start_value: float,
    peak_value: float,
    final_value: float,
    epochs: int,
    n_batches_per_epoch: int,
    warmup_epochs: int = 0,
) -> List[float]:
    """Cosine scheduler with optional linear warmup

    Parameters
    ----------
    start_value : float
        Initial value of the scheduler
    peak_value : float
        Peak value of the scheduler; i.e. the maximum learning rate at
        the end of the linear warmup phase
    final_value : float
        Final value of the scheduler
    epochs : int
        Number of epochs in training run
    n_batches_per_epoch : int
        Number of batches iterated over during each epoch
    warmup_epochs : int
        Number of epochs to perform the linear warmup

    Notes
    -----
    The original DINO paper uses cosine scheduling for the EMA of
    both the learning rate and the weight decay of the optimizer
    during training.
    """

    assert warmup_epochs < epochs, \
        "Warmup epochs must be < total epochs"

    warmup_schedule = []
    warmup_steps = warmup_epochs * n_batches_per_epoch
    for i in range(warmup_steps):
        warmup_schedule.append(
            start_value + (peak_value - start_value) * i / warmup_steps
        )

    cosine_schedule = []
    cosine_steps = epochs * n_batches_per_epoch - warmup_steps
    for i in range(cosine_steps):
        cosine_schedule.append(
            final_value + 0.5 * (peak_value - final_value)
            * (1 + torch.cos(torch.tensor(torch.pi * i / cosine_steps)).item())
        )

    schedule = warmup_schedule + cosine_schedule

    assert len(schedule) == warmup_steps + cosine_steps, \
        "Schedule length must equal total number of steps"

    return schedule

def clip_gradients(
    model: nn.Module,
    clip: float = 3.0
) -> List[float]:
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


# -----------------------------------------------------------------------------
#                                   Training
# -----------------------------------------------------------------------------

from torchvision import datasets
from torch.utils.data import DataLoader

from backbones.mlp_mixer import MLPMixer

image_folder = '/Users/tomouellette/Home/software/aloe/output/dino_mnist/plots'
batch_size = 2

image_size = 224
channels = 3

projector_hidden_dim = 256
projector_k = 256
projector_layers = 4
projector_batch_norm = False
projector_l2_norm = False

global_crops_scale = (0.5, 1.0)
local_crops_scale = (0.3, 0.7)
local_crops_number = 8


transform = Augmentation(
    image_size,
    global_crops_scale,
    local_crops_scale,
    local_crops_number,
)

dataset = datasets.ImageFolder(image_folder, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

backbone = MLPMixer(
    image_size,
    channels,
    patch_size=16,
    dim=64,
    depth=4,
    n_classes=1000,
    expansion_factor=4,
    expansion_factor_token=0.5,
    dropout=0.
)

backbone.head = nn.Identity()

model = StudentTeacher(
    backbone,
    image_size,
    projector_hidden_dim,
    projector_k,
    projector_layers,
    projector_batch_norm,
    projector_l2_norm,
)
# 
# 
# def backbone(x): return x
# 
# 
# student = BackboneProjector(
#     backbone,
#     (channels, image_size, image_size),
#     projector_hidden_dim,
#     projector_k,
#     projector_layers,
#     projector_batch_norm,
#     projector_l2_norm
# )
# 
# teacher = BackboneProjector(
#     copy.deepcopy(backbone),
#     (channels, image_size, image_size),
#     projector_hidden_dim,
#     projector_k,
#     projector_layers,
#     projector_batch_norm,
#     False
# )


# accelerator = Accelerator()
# dataloader, model, optimizer, scheduler = accelerator.prepare(
#         dataloader, model, optimizer, scheduler
# )
#   
# device = accelerator.device
# 
# for batch in dataloader:
#     inputs, targets = batch
#     inputs = inputs.to(device)
#     targets = targets.to(device)
#     outputs = model(inputs)
#     loss = loss_function(outputs, targets)
#     loss.backward()
#     accelerator.backward(loss)
#     optimizer.step()
#     scheduler.step()
#     optimizer.zero_grad()

