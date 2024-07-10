import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distillation with no labels")

    parser.add_argument(
        "--image_size", type=int, default=28,
        help="Input image size (assumes square images)"
    )

    parser.add_argument(
        "--channels", type=int, default=1,
        help="Number of input channels"
    )

    parser.add_argument(
        "--projector_hidden_dim", type=int, default=256,
        help="Dimension of hidden layers in projector network"
    )

    parser.add_argument(
        "--projector_k", type=int, default=4096,
        help="Output dimension of projector network (i.e. prototype count)"
    )

    parser.add_argument(
        "--projector_layers", type=int, default=4,
        help="Number of layers in projector network"
    )

    parser.add_argument(
        "--projector_batch_norm", type=bool, default=False,
        help="Use batch normalization in projector network"
    )

    parser.add_argument(
        "--projector_l2_norm", type=bool, default=False,
        help="Use L2 normalization in projector network"
    )

    parser.add_argument(
        "--t_teacher", type=float, default=0.04,
        help="Temperature of teacher network"
    )

    parser.add_argument(
        "--t_student", type=float, default=0.9,
        help="Temperature of student network"
    )

    parser.add_argument(
        "--crop_local_scales", type=float, default=(0.5, 1.0),
        help="Local crop scales",
        nargs="+"
    )

    parser.add_argument(
        "--crop_global_scales", type=float, default=(0.5, 1.0),
        help="Global crop scales",
        nargs="+"
    )

    parser.add_argument(
        "--ema_decay_teacher", type=float, default=0.99,
        help="Exponential moving average decay of teacher network"
    )

    parser.add_argument(
        "--ema_decay_center", type=float, default=0.9,
        help="Exponential moving average decay of center"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size"
    )

    parser.add_argument(
        "--epochs", type=int, default=24,
        help="Number of epochs"
    )

    parser.add_argument(
        "--lr_max", type=float, default=0.0005,
        help="Maximum learning rate"
    )

    parser.add_argument(
        "--lr_min", type=float, default=0.000001,
        help="Minimum learning rate"
    )

    parser.add_argument(
        "--lr_warmup", type=int, default=10,
        help="Number of warmup epochs for cosine annealing learning rate"
    )

    parser.add_argument(
        "--weight_decay_start", type=float, default=0.04,
        help="Weight decay of Adam optimizer at start of training"
    )

    parser.add_argument(
        "--weight_decay_end", type=float, default=0.4,
        help="Weight decay of Adam optimizer at end of training"
    )

    parser.add_argument(
        "--clip_grad", type=float, default=0.,
        help="Gradient clipping of backbone during training"
    )

    parser.add_argument(
        "--freeze_projector", type=int, default=1,
        help="Number of epochs to freeze the projector"
    )

    parser.add_argument(
        "--anneal_momentum", type=bool, default=True,
        help="Anneal momentum to 1.0 during training"
    )

    parser.add_argument(
        "--t_teacher_final", type=float, default=None,
        help="Final temperature of teacher network"
    )

    parser.add_argument(
        "--t_teacher_warmup", type=float, default=None,
        help="Initial temperature of teacher network"
    )

    parser.add_argument(
        "--t_teacher_warmup_epochs", type=int, default=None,
        help="Number of epochs to warmup teacher network"
    )

    parser.add_argument(
        "--n_train", type=int, default=None,
        help="Number of training samples"
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for training"
    )

    return parser.parse_args()
