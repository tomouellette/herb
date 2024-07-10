#!/usr/bin/env python3
import os
import torch
import shutil
import random
import warnings
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from bin.augment import basic_augment
from bin.dino import DINO, DINOTrainer
from bin.mlp_mixer import MLPMixer
from bin.mlp import MLP
from bin.utils import message, generate_gif

from args.dino import parse_args

_OUTPUT_ = "output/dino_mnist/"


def example_data(batch_size: int = 256) -> tuple:
    """Load MNIST data"""
    import ssl
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    ssl._create_default_https_context = ssl._create_unverified_context

    train = MNIST(root="data", train=True, download=True, transform=ToTensor())
    test = MNIST(root="data", train=False, download=True, transform=ToTensor())

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def example_backbone() -> MLPMixer:
    """Build an example MLP-Mixer backbone"""
    #backbone = MLPMixer(
    #    img_size=28,
    #    in_chans=1,
    #    patch_size=4,
    #    dim=512,
    #    depth=2,
    #    expansion_factor=2,
    #    expansion_factor_token=2,
    #    dropout=0.,
    #    num_classes=1,
    #)

    #backbone.head = torch.nn.Identity()

    backbone = MLP(
        input_dim=28 * 28,
        hidden_layers=[1024, 512, 256],
        output_dim=256,
        dropout_rate=0.05,
        activation=torch.nn.Tanh
    )

    return backbone


def example_manifold(
    model: DINO,
    test_loader: torch.utils.data.DataLoader,
    device: str
):
    tz, sz, y = [], [], []
    for xi, yi in tqdm(test_loader):
        tzi = model.embedding(xi, device=device, mode="teacher")
        szi = model.embedding(xi, device=device, mode="student")
        tz.append(tzi.detach().cpu().numpy())
        sz.append(szi.detach().cpu().numpy())
        y.append(yi.numpy())

    tz = np.concatenate(tz, axis=0)
    sz = np.concatenate(sz, axis=0)
    y = np.concatenate(y, axis=0)

    tsne = TSNE(n_components=2, n_jobs=-1)
    tz_tsne = tsne.fit_transform(tz)
    sz_tsne = tsne.fit_transform(sz)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(*tz_tsne.T, c=y, cmap="tab10", alpha=0.3, s=2)
    ax[1].scatter(*sz_tsne.T, c=y, cmap="tab10", alpha=0.3, s=2)

    ax[0].set_title("Teacher")
    ax[1].set_title("Student")

    ax[0].axis("off")
    ax[1].axis("off")

    fig.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_, "tsne.png"))


def main(args: argparse.Namespace):
    """Train DINO on MNIST using an MLP-Mixer backbone"""

    torch.manual_seed(123456)
    np.random.seed(123456)
    random.seed(123456)

    backbone = example_backbone()

    if os.path.exists(_OUTPUT_):
        shutil.rmtree(_OUTPUT_)

    os.makedirs(os.path.join(_OUTPUT_, "plots"), exist_ok=True)

    model = DINO(
        backbone,
        image_size=(args.channels, args.image_size, args.image_size),
        projector_hidden_dim=args.projector_hidden_dim,
        projector_k=args.projector_k,
        projector_layers=args.projector_layers,
        projector_batch_norm=args.projector_batch_norm,
        projector_l2_norm=args.projector_l2_norm,
        t_teacher=args.t_teacher,
        t_student=args.t_student,
        crop_local_scales=tuple(args.crop_local_scales),
        crop_global_scales=tuple(args.crop_global_scales),
        ema_decay_teacher=args.ema_decay_teacher,
        ema_decay_center=args.ema_decay_center,
        t_augment=basic_augment(),
        s_augment=basic_augment(),
    )

    train_loader, test_loader = example_data(args.batch_size)

    trainer = DINOTrainer(
        model,
        train_loader,
        loader_batch_size=args.batch_size,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        lr_warmup=args.lr_warmup,
        epochs=args.epochs,
        weight_decay_start=args.weight_decay_start,
        weight_decay_end=args.weight_decay_end,
        clip_grad=args.clip_grad,
        anneal_momentum=args.anneal_momentum,
        freeze_projector=args.freeze_projector,
        t_teacher_final=args.t_teacher_final,
        t_teacher_warmup=args.t_teacher_warmup,
        t_teacher_warmup_epochs=args.t_teacher_warmup_epochs,
        n_train=args.n_train,
    )

    post_epoch_logger = {'epoch': [], 'teacher': [], 'student': []}

    @torch.no_grad()
    def example_post_epoch(model: DINO, epoch: int, device: str):
        """Post-epoch callback for evaluating DINO training"""
        model.eval()
        tz, sz, y = [], [], []
        for xi, yi in tqdm(test_loader):
            tzi = model.embedding(xi, device=device, mode="teacher")
            szi = model.embedding(xi, device=device, mode="student")
            tz.append(tzi.detach().cpu().numpy())
            sz.append(szi.detach().cpu().numpy())
            y.append(yi.numpy())

        tz = np.concatenate(tz, axis=0)
        sz = np.concatenate(sz, axis=0)
        y = np.concatenate(y, axis=0)

        tz = (tz - tz.mean(axis=0)) / tz.std(axis=0)
        sz = (sz - sz.mean(axis=0)) / sz.std(axis=0)

        z_pca = PCA(n_components=2)
        tz_pca = z_pca.fit_transform(tz)

        z_pca = PCA(n_components=2)
        sz_pca = z_pca.fit_transform(sz)

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))

        ax[0].scatter(*tz_pca.T, c=y, cmap="tab10", alpha=0.5, s=1.75)
        ax[1].scatter(*sz_pca.T, c=y, cmap="tab10", alpha=0.5, s=1.75)

        ax[0].set_title("Teacher")
        ax[1].set_title("Student")

        for i in range(2):
            ax[i].get_yaxis().set_visible(False)
            ax[i].get_xaxis().set_visible(False)
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])
            ax[i].set_yticks([])
            ax[i].set_xticks([])
            ax[i].set_xlabel(None)
            ax[i].set_ylabel(None)

        fig.tight_layout()
        plt.savefig(os.path.join(_OUTPUT_, "plots", f"epoch_{epoch}.png"))
        plt.close()

        if epoch != 0 and epoch % 10 == 0:
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
                message("DINO", f"Epoch {epoch} {name} accuracy: {acc}")

                post_epoch_logger['epoch'].append(epoch)
                post_epoch_logger[name].append(acc)

    trainer.fit(
        device=args.device,
        post_epoch_callable=example_post_epoch
    )

    image_paths = [
        os.path.join(_OUTPUT_, "plots", f"epoch_{i}.png")
        for i in range(args.epochs)
    ]

    generate_gif(image_paths, os.path.join(_OUTPUT_, "train_dino_mnist.gif"))
    example_manifold(model, test_loader, args.device)


if __name__ == "__main__":
    main(parse_args())
