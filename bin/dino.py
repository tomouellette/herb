import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from torchvision import transforms as T
from typing import Optional, Callable, Union
from torch.utils.data import DataLoader

from .augment import basic_augment
from .utils import cosine_scheduler, clip_gradients, trunc_normal_
from .utils import message


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
        self.apply(self._init_weights)
        self.head = nn.Linear(projector_hidden_dim, projector_k)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x.flatten(1))
        if self.l2_norm:
            norm = x.norm(dim=1, keepdim=True).clamp(min=1e-6)
            x = x / norm

        return self.head(x)



class DINO(nn.Module):
    """Self-supervised training with distillation with no labels

    Parameters
    ----------
    backbone : nn.Module
        Backbone model for generting embeddings
    image_size : tuple[int, int, int]
        Image size in (C, H, W)
    projector_hidden_dim : int
        Hidden dimension for projectors
    projector_k : int
        Output dimension for projectors (i.e. prototypes)
    projector_layers : int
        Number of layers in the projectors
    projector_batch_norm : bool
        Whether to use batch normalization in projectors
    projector_l2_norm : bool
        Whether to l2-normalize the final output of projectors
    t_teacher : float
        Initial temperature for teacher network
    t_student : float
        Initial temperature for student network
    crop_local_scales : tuple[float, float]
        Crop scales for generating local views
    crop_global_scales : tuple[float, float]
        Crop scales for generating global views
    ema_decay_teacher : float
        Exponential moving average decay for teacher network
    ema_decay_center : float
        Exponential moving average decay for center
    t_augment : Optional[Callable]
        Augmentation function for teacher network image copy
    s_augment : Optional[Callable]
        Augmentation function for student network image copy
    """

    def __init__(
        self,
        backbone: nn.Module,
        image_size: tuple[int, int, int],
        projector_hidden_dim: int = 256,
        projector_k: int = 256,
        projector_layers: int = 4,
        projector_batch_norm: bool = False,
        projector_l2_norm: bool = False,
        t_teacher: float = 0.03,
        t_student: float = 0.9,
        crop_local_scales: tuple[float, float] = (0.6, 1.0),
        crop_global_scales: tuple[float, float] = (0.5, 1.0),
        ema_decay_teacher: float = 0.9,
        ema_decay_center: float = 0.9,
        t_augment: Optional[Callable] = basic_augment(),
        s_augment: Optional[Callable] = basic_augment(),
    ):
        super().__init__()
        self.c, self.h, self.w = image_size
        self.t_teacher = t_teacher
        self.t_student = t_student
        self.ema_decay_teacher = ema_decay_teacher
        self.ema_decay_center = ema_decay_center

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
        self.teacher_projector = copy.deepcopy(self.student_projector)

        self.current_teacher_center = nn.Parameter(
            torch.zeros(1, projector_k),
            requires_grad=False
        )

        self.previous_teacher_center = nn.Parameter(
            torch.zeros(1, projector_k),
            requires_grad=False
        )

        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)

        for p in self.teacher_projector.parameters():
            p.requires_grad_(False)

        self.t_augment = t_augment
        self.s_augment = s_augment

        if crop_local_scales is None:
            self.local_crop = nn.Identity()
        else:
            self.local_crop = T.RandomResizedCrop(
                (self.h, self.w),
                scale=crop_local_scales
            )

        if crop_global_scales is None:
            self.global_crop = nn.Identity()
        else:
            self.global_crop = T.RandomResizedCrop(
                (self.h, self.w),
                scale=crop_global_scales
            )

    @torch.no_grad()
    def forward_ema(self, beta: float) -> None:
        for pt, ps in zip(
            self.teacher_encoder.parameters(),
            self.student_encoder.parameters()
        ):
            pt.data = pt.data * beta + ps.data * (1.0 - beta)

        for pt, ps in zip(
            self.teacher_projector.parameters(),
            self.student_projector.parameters()
        ):
            pt.data = pt.data * beta + ps.data * (1.0 - beta)

        updated_centers = self.current_teacher_center * beta \
            + self.previous_teacher_center * (1.0 - beta)

        self.current_teacher_center.copy_(updated_centers)

    def forward_device(self, device: Union[str, torch.device]):
        self.to(device)
        self.student_encoder.to(device)
        self.student_projector.to(device)
        self.teacher_encoder.to(device)
        self.teacher_projector.to(device)

    def forward_center(self, p1_teacher: Tensor, p2_teacher: Tensor):
        teacher_avg = torch.cat((p1_teacher, p2_teacher))
        teacher_avg = teacher_avg.mean(dim=0, keepdim=True)
        self.previous_teacher_center.copy_(teacher_avg)

    def forward_loss(
        self,
        p1_teacher: Tensor,
        p2_teacher: Tensor,
        p1_student: Tensor,
        p2_student: Tensor
    ) -> Tensor:
        p1_teacher = p1_teacher.detach()
        p2_teacher = p2_teacher.detach()

        p1_teacher = (p1_teacher - self.current_teacher_center)
        p1_teacher /= self.t_teacher

        p2_teacher = (p2_teacher - self.current_teacher_center)
        p2_teacher /= self.t_teacher

        p1_teacher = p1_teacher.softmax(dim=-1)
        p2_teacher = p2_teacher.softmax(dim=-1)

        p1_student = p1_student / self.t_student
        p2_student = p2_student / self.t_student

        p1_student = F.log_softmax(p1_student + 1e-20, dim=-1)
        p2_student = F.log_softmax(p2_student + 1e-20, dim=-1)

        l1 = -(p1_teacher * p2_student).sum(dim=-1).mean()
        l2 = -(p2_teacher * p1_student).sum(dim=-1).mean()

        return (l1 + l2) / 2

    @torch.no_grad()
    def embedding(
        self,
        x: torch.Tensor,
        device: Union[str, torch.device] = "cpu",
        mode: str = "student"
    ) -> torch.Tensor:
        x = x.to(device)
        self.forward_device(device)

        if mode == "student":
            return self.student_projector(self.student_encoder(x))
        elif mode == "teacher":
            return self.teacher_projector(self.teacher_encoder(x))
        else:
            raise ValueError("Mode must be either 'student' or 'teacher'")

    def forward(
        self,
        x: torch.Tensor,
        beta: Optional[float] = None,
        t_teacher: Optional[float] = None,
        t_student: Optional[float] = None,
        device: Union[str, torch.device] = "cpu"
    ):
        self.t_teacher = t_teacher if t_teacher is not None else self.t_teacher
        self.t_student = t_student if t_student is not None else self.t_student

        beta = beta if beta is not None else self.ema_decay_teacher

        self.forward_ema(beta)

        x1 = self.t_augment(x)
        x2 = self.s_augment(x)

        x1_local = self.local_crop(x1).to(device)
        x2_local = self.local_crop(x2).to(device)

        x1_global = self.global_crop(x1).to(device)
        x2_global = self.global_crop(x2).to(device)

        self.forward_device(device)

        s1 = self.student_encoder(x1_local)
        s2 = self.student_encoder(x2_local)

        p1_student = self.student_projector(s1)
        p2_student = self.student_projector(s2)

        with torch.no_grad():
            t1 = self.teacher_encoder(x1_global)
            t2 = self.teacher_encoder(x2_global)

            p1_teacher = self.teacher_projector(t1)
            p2_teacher = self.teacher_projector(t2)

        self.forward_center(p1_teacher, p2_teacher)

        loss = self.forward_loss(
            p1_teacher=p1_teacher,
            p2_teacher=p2_teacher,
            p1_student=p1_student,
            p2_student=p2_student
        )

        return loss


class DINOTrainer:
    """A basic trainer for distillation with no labels

    Parameters
    ----------
    model : DINO
        A initialized DINO model
    loader : DataLoader
        A PyTorch DataLoader object
    loader_batch_size : int
        Batch size for the DataLoader
    lr_max : float
        Maximum learning rate
    lr_min : float
        Minimum learning rate
    lr_warmup : int
        Number of warmup epochs for learning rate
    epochs : int
        Number of epochs to train
    weight_decay_start : float
        Initial Adam weight decay value
    weight_decay_end : float
        Final Adam weight decay value
    clip_grad : float
        If greater than 0, clip gradients in student network
    anneal_momentum : bool
        Whether to anneal momentum in teacher network to 1.0
        over the course of training
    freeze_projector : int
        Number of epochs to freeze the projector gradients prior to
        updating weights
    t_teacher_warmup : Optional[float]
        Initial temperature for teacher network
    t_teacher_warmup_epochs : Optional[int]
        Number of warmup epochs for teacher network temperature
    n_train : Optional[int]
        Number of batches per epoch. If loader does not have a __len__ method,
        then this value must be provided.
    """

    def __init__(
        self,
        model: DINO,
        loader: DataLoader,
        loader_batch_size: int = 32,
        lr_max: float = 0.0005,
        lr_min: float = 0.0001,
        lr_warmup: int = 10,
        epochs: int = 32,
        weight_decay_start: float = 0.04,
        weight_decay_end: float = 0.4,
        clip_grad: float = 3.0,
        anneal_momentum: bool = True,
        freeze_projector: int = 1,
        t_teacher_final: Optional[float] = None,
        t_teacher_warmup: Optional[float] = None,
        t_teacher_warmup_epochs: Optional[int] = None,
        n_train: Optional[int] = None
    ):
        self.settings = {
            k: v for k, v in locals().items()
            if k not in ["self", "model", "loader_train", "loader_test"]
        }

        self.model = model
        self.epochs = epochs
        self.clip_grad = clip_grad
        self.freeze_projector = freeze_projector
        self.loader = loader

        if n_train is None:
            self.n_train = len(loader)
        else:
            self.n_train = n_train / loader_batch_size

        self.schedule_lr = cosine_scheduler(
            lr_max * loader_batch_size / 256,
            lr_min,
            epochs,
            self.n_train,
            warmup_epochs=lr_warmup,
            start_warmup_value=0
        )

        self.schedule_weight_decay = cosine_scheduler(
            weight_decay_start,
            weight_decay_end,
            epochs,
            self.n_train,
            warmup_epochs=0,
        )

        if anneal_momentum:
            self.schedule_ema_beta = cosine_scheduler(
                model.ema_decay_teacher,
                1.0,
                epochs,
                self.n_train,
                warmup_epochs=0
            )
        else:
            self.schedule_ema_beta = np.array(
                [model.ema_decay_teacher] * epochs * self.n_train
            )

        if np.all([
            t_teacher_warmup is not None,
            t_teacher_warmup_epochs is not None,
            t_teacher_final is not None
        ]):
            self.schedule_teacher_temp = np.linspace(
                t_teacher_warmup,
                model.t_teacher,
                t_teacher_warmup_epochs
            )

            self.schedule_teacher_temp = np.concatenate((
                self.schedule_teacher_temp,
                np.array(
                    [t_teacher_final] * (epochs - t_teacher_warmup_epochs)
                )
            ))
        else:
            self.schedule_teacher_temp = np.array([model.t_teacher] * epochs)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.schedule_lr[0],
            weight_decay=self.schedule_weight_decay[0]
        )

        self.logger = {}

    def train_step(
        self,
        images: torch.Tensor,
        epoch: int,
        batch_idx: int,
        device: Union[str, torch.device] = "cpu"
    ):
        idx = epoch * self.n_train + batch_idx
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.schedule_lr[idx]
            if i == 0:
                param_group["weight_decay"] = self.schedule_weight_decay[idx]

        loss = self.model(
            images,
            device=device,
            beta=self.schedule_ema_beta[idx],
            t_teacher=self.schedule_teacher_temp[epoch],
        )

        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_grad > 0.:
            _ = clip_gradients(
                self.model,
                self.clip_grad
            )

        if epoch < self.freeze_projector:
            for p in self.model.student_projector.parameters():
                p.requires_grad_(False)

        self.optimizer.step()

        self.logger[epoch]["lr"].append(
            self.schedule_lr[idx]
        )

        self.logger[epoch]["wd"].append(
            self.schedule_weight_decay[idx]
        )

        self.logger[epoch]["loss"].append(
            loss.item()
        )

        return loss.item()

    def fit(
        self,
        device: str = "cpu",
        post_epoch_callable: Callable = None,
        silent: bool = False,
    ):
        if not silent:
            message("DINO", "Training started")

        for epoch in range(self.epochs):

            self.model.student_encoder.train()
            self.model.student_projector.train()

            with tqdm(self.loader, disable=silent) as progress:

                self.logger[epoch] = {"lr": [], "wd": [], "loss": []}

                for batch_idx, (images, *_) in enumerate(progress):
                    loss = self.train_step(images, epoch, batch_idx, device)
                    progress.set_description(f"Loss: {loss}")

            loss = np.mean(self.logger[epoch]["loss"])

            if not silent:
                message("DINO", f"Epoch {epoch} | Loss: {loss}")

            if post_epoch_callable is not None:
                post_epoch_callable(self.model, epoch, device)
