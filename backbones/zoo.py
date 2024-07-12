import torch
import torch.nn as nn


def select_backbone(model: str) -> nn.Module:
    if model == "vit_tiny":
        from vit import vit_tiny
        return vit_tiny()
    elif model == "vit_small":
        from vit import vit_small
        return vit_small()
    elif model == "vit_base":
        from vit import vit_base
        return vit_base()
    elif model == "vit_large":
        from vit import vit_large
        return vit_large()
    elif model == "mlp_mixer_tiny":
        from mlp_mixer import mlp_mixer_tiny
        return mlp_mixer_tiny()
    elif model == "mlp_mixer_small":
        from mlp_mixer import mlp_mixer_small
        return mlp_mixer_small()
    elif model == "mlp_mixer_base":
        from mlp_mixer import mlp_mixer_base
        return mlp_mixer_base()
    elif model == "mlp_mixer_large":
        from mlp_mixer import mlp_mixer_large
        return mlp_mixer_large()
    elif model == "navit_tiny":
        from navit import navit_tiny
        return navit_tiny()
    elif model == "navit_small":
        from navit import navit_small
        return navit_small()
    elif model == "navit_base":
        from navit import navit_base
        return navit_base()
    elif model == "navit_large":
        from navit import navit_large
        return navit_large()
    else:
        raise ValueError(f"Unknown model: {model}")


if __name__ == "__main__":
    prefix = "[INFO | zoo ]"

    print(f"{prefix} Checking zoo model loading.")

    models = [
        "vit_tiny",
        "vit_small",
        "vit_base",
        "vit_large",
        "mlp_mixer_tiny",
        "mlp_mixer_small",
        "mlp_mixer_base",
        "mlp_mixer_large",
        "navit_tiny",
        "navit_small",
        "navit_base",
        "navit_large",
    ]

    for model in models:
        print(f"{prefix} Loading {model}")
        model = select_backbone(model)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"{prefix} {model.__class__.__name__}")

    print(f"{prefix} Passed zoo basic checks.")
