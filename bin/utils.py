import os
import sys
import math
import torch
import imageio
import datetime
import warnings
import numpy as np
import torch.nn as nn
from typing import List


def initialize():
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../')
    ))


def message(model: str, output: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO | {timestamp} | {model}] {output}")


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    iters_per_epoch: int,
    warmup_epochs=0,
    start_warmup_value=0
) -> np.ndarray:
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * iters_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value,
            base_value,
            warmup_iters
        )
    iters = np.arange(epochs * iters_per_epoch - warmup_iters)
    schedule = final_value + 0.5 \
        * (base_value - final_value) \
        * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * iters_per_epoch
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


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Source: pytorch official master
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " +
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        ll = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * ll - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def topk_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[:k].reshape(-1).float().sum(0) * 100. / batch_size
        for k in topk
    ]


def generate_gif(ordered_image_paths: list, output: str) -> None:
    images = []
    for filename in ordered_image_paths:
        images.append(imageio.v3.imread(filename))

    for _ in range(20):
        images.append(imageio.v3.imread(ordered_image_paths[-1]))

    imageio.v3.imwrite(
        output,
        images,
        loop=1000
    )
