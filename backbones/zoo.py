import torch
import torch.nn as nn


def select_backbone(
    model: str,
    image_size: int,
    channels: int,
    **kwargs
) -> nn.Module:
    kwargs = {"image_size": image_size, "channels": channels, **kwargs}

    if model == "mlp_nano":
        from backbones.mlp import mlp_nano
        return mlp_nano(**kwargs)
    elif model == "mlp_micro":
        from backbones.mlp import mlp_micro
        return mlp_micro(**kwargs)
    elif model == "mlp_tiny":
        from backbones.mlp import mlp_tiny
        return mlp_tiny(**kwargs)
    elif model == "mlp_small":
        from backbones.mlp import mlp_small
        return mlp_small(**kwargs)
    elif model == "mlp_base":
        from backbones.mlp import mlp_base
        return mlp_base(**kwargs)
    elif model == "mlp_large":
        from backbones.mlp import mlp_large
        return mlp_large(**kwargs)
    elif model == "convnext_nano":
        from backbones.convnext import convnext_nano
        return convnext_nano(**kwargs)
    elif model == "convnext_micro":
        from backbones.convnext import convnext_micro
        return convnext_micro(**kwargs)
    elif model == "convnext_tiny":
        from backbones.convnext import convnext_tiny
        return convnext_tiny(**kwargs)
    elif model == "convnext_small":
        from backbones.convnext import convnext_small
        return convnext_small(**kwargs)
    elif model == "convnext_base":
        from backbones.convnext import convnext_base
        return convnext_base(**kwargs)
    elif model == "convnext_large":
        from backbones.convnext import convnext_large
        return convnext_large(**kwargs)
    elif model == 'vit_nano':
        from backbones.vit import vit_nano
        return vit_nano(**kwargs)
    elif model == "vit_micro":
        from backbones.vit import vit_micro
        return vit_micro(**kwargs)
    if model == "vit_tiny":
        from backbones.vit import vit_tiny
        return vit_tiny(**kwargs)
    elif model == "vit_small":
        from backbones.vit import vit_small
        return vit_small(**kwargs)
    elif model == "vit_base":
        from backbones.vit import vit_base
        return vit_base(**kwargs)
    elif model == "vit_large":
        from backbones.vit import vit_large
        return vit_large(**kwargs)
    elif model == "mlp_mixer_nano":
        from backbones.mlp_mixer import mlp_mixer_nano
        return mlp_mixer_nano(**kwargs)
    elif model == "mlp_mixer_micro":
        from backbones.mlp_mixer import mlp_mixer_micro
        return mlp_mixer_micro(**kwargs)
    elif model == "mlp_mixer_tiny":
        from backbones.mlp_mixer import mlp_mixer_tiny
        return mlp_mixer_tiny(**kwargs)
    elif model == "mlp_mixer_small":
        from backbones.mlp_mixer import mlp_mixer_small
        return mlp_mixer_small(**kwargs)
    elif model == "mlp_mixer_base":
        from backbones.mlp_mixer import mlp_mixer_base
        return mlp_mixer_base(**kwargs)
    elif model == "mlp_mixer_large":
        from backbones.mlp_mixer import mlp_mixer_large
        return mlp_mixer_large(**kwargs)
    elif model == "navit_nano":
        from backbones.navit import navit_nano
        return navit_nano(**kwargs)
    elif model == "navit_micro":
        from backbones.navit import navit_micro
        return navit_micro(**kwargs)
    elif model == "navit_tiny":
        from backbones.navit import navit_tiny
        return navit_tiny(**kwargs)
    elif model == "navit_small":
        from backbones.navit import navit_small
        return navit_small(**kwargs)
    elif model == "navit_base":
        from backbones.navit import navit_base
        return navit_base(**kwargs)
    elif model == "navit_large":
        from backbones.navit import navit_large
        return navit_large(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model}")


def available_backbones():
    return [
        "mlp_nano",
        "mlp_micro",
        "mlp_tiny",
        "mlp_small",
        "mlp_base",
        "mlp_large",
        "convnext_nano",
        "convnext_micro",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
        "vit_nano",
        "vit_micro",
        "vit_tiny",
        "vit_small",
        "vit_base",
        "vit_large",
        "mlp_mixer_nano",
        "mlp_mixer_micro",
        "mlp_mixer_tiny",
        "mlp_mixer_small",
        "mlp_mixer_base",
        "mlp_mixer_large",
        "navit_nano",
        "navit_micro",
        "navit_tiny",
        "navit_small",
        "navit_base",
        "navit_large",
    ]


if __name__ == "__main__":
    prefix = "[INFO | zoo ]"

    print(f"{prefix} Checking zoo model loading.")

    models = available_backbones()

    for model in models:
        print(f"{prefix} Loading {model}")
        model = select_backbone(model, image_size=224, channels=3)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"{prefix} {model.__class__.__name__}")

    print(f"{prefix} Passed zoo basic checks.")
