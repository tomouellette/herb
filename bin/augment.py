import random
import torch
import torch.nn as nn
from torchvision import transforms as T
from typing import Callable


class Randomize(nn.Module):
    def __init__(self, f: Callable, p: float):
        super().__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.f(x)


def basic_augment() -> Callable:
    return torch.nn.Sequential(
        Randomize(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
        Randomize(T.RandomRotation(degrees=(-60, 60)), p=0.3),
        Randomize(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.3),
    )
