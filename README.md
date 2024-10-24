# :herb: herb

A rough collection of personal minimalistic scripts, utilities, and implementations for experimenting, building, and training neural networks. Backbones and models are all self-contained in single files for easy modification or extension in more elaborate downstream projects.

## Setup

It's assumed you have `python3` and `pip` installed. You can set up the required dependencies as shown below.

```bash
git clone https://github.com/tomouellette/herb
cd herb/
python3 -m pip install -r requirements.txt
```

By default, all I/O for various models is handled through a basic `FolderDataset`. If you want faster I/O that works across shards, you can replace these blocks with a `webdataset` dataset if you'd like.

## Backbones

Backbones are pre-specified as `nano`, `micro`, `tiny`, `small`, `base`, and `large` with approximately 2.5M, 5M, 10M, 20M, 80M, and 200M+ parameters at `3 x 224 x 224` image resolution, respectively. Note that the basic `MLP` does not follow this parameter scaling. You can check that each implementation is working (and the parameters) as follows:

```bash
python3 backbones/convnext.py
python3 backbones/mlp.py
python3 backbones/mlp_mixer.py
python3 backbones/navit.py
python3 backbones/vit.py
```

You can also validate that the entire zoo of models is working on your machine.

```bash
python3 -m backbones.zoo
```

If you run a backbone interactively or in a script, you can save them to `pytorch` or `safetensors` format as follows:

```python3
from backbones.mlp_mixer import mlp_mixer_small

model = mlp_mixer_small()
model.save("mlp_mixer_small.pth") # PyTorch format
model.save("mlp_mixer_small.safetensors") # safetensors format
```

## Models

Models encompass general training or pre-training schemes for various tasks. Everything is setup for the single GPU setting but can be easily modified for multi-GPU. You can check that these models are working on your machine by running the following code.

```bash
chmod +x models/test.sh
./models/test.sh
```

### DINO

A trainable implementation of [distillation with no labels (DINO)](https://arxiv.org/abs/2104.14294). This can be trained on a folder full of arbitrarily sized images.

```bash
python3 -m models.dino \
    --input image_folder/ \
    --output logs/ \
    --image_size 224 \
    --channels 3 \
    --backbone vit_small \
    --projector_hidden_dim 256 \
    --projector_k 256 \
    --projector_layers 4 \
    --projector_batch_norm False \
    --projector_l2_norm False \
    --momentum_center 0.9 \
    --momentum_teacher 0.996 \
    --global_crops_scale 0.5 1.0 \
    --local_crops_scale 0.3 0.7 \
    --n_views 6 \
    --t_student 0.1 \
    --t_teacher 0.04 \
    --epochs 512 \
    --batch_size 256 \
    --num_workers 4 \
    --n_batches 1000 \
    --lr_max 1e-4 \
    --lr_min 1e-6 \
    --lr_warmup 0.1 \
    --weight_decay 0.05 \
    --n_checkpoint 10 \
    --print_fraction 0.025
```

### Masked Autoencoder (MAE)

A trainable implementation of [masked autoencoder](https://arxiv.org/abs/2111.06377). This can be trained on a folder full of arbitrarily sized images.

```bash
python3 -m models.mae \
    --input image_folder/ \
    --output logs/ \
    --backbone vit_small \
    --image_size 224 \
    --channels 3 \
    --patch_size 16 \
    --mask_ratio 0.7 \
    --batch_size 256 \
    --num_workers 4 \
    --n_batches 1000 \
    --epochs 512 \
    --lr_min 1e-6 \
    --lr_max 0.001 \
    --weight_decay 1e-6 \
    --lr_warmup 0.1 \
    --n_checkpoint 10 \
    --print_fraction 0.025
```

### Masked Barlow Twins (MBT)

A custom variant of [barlow twins](https://arxiv.org/pdf/2103.03230) with additional token masking and attention pooling of embedded views. This can be trained on a folder full of arbitrarily sized images.

```bash
python3 -m models.mbt \
    --input image_folder/ \
    --output logs/ \
    --backbone vit_small \
    --image_size 224 \
    --channels 3 \
    --patch_size 16 \
    --mask_ratio_min 0.3 \
    --mask_ratio_max 0.3 \
    --rr_lambda 0.0051 \
    --projector_dims 512 512 2048 \
    --n_views 2 \
    --batch_size 64 \
    --num_workers 4 \
    --epochs 512 \
    --lr_min 1e-6 \
    --lr_max 0.001 \
    --weight_decay 1e-6 \
    --lr_warmup 0.1 \
    --n_checkpoint 10 \
    --print_fraction 0.025
```

## Inference

For a variety of backbones, I've also written matching implementations in Rust using the `candle` library. Any backbone trained in pytorch and saved as a safetensor can be loaded and run in Rust. You can paste/modify the code into your project as needed. Here's some example backbones, assuming you have `cargo` installed.

### ConvNext

```bash
cd backbones/candle_convnext/
cargo run --release # >> ConvNext Output: [1, 1000]
```

### MLPMixer

```bash
cd backbones/candle_mlp_mixer/
cargo run --release # >> MLPMixer Output: [1, 1000]
```

### ViT

```bash
cd backbones/candle_vit/
cargo run --release # >> ViT Output: [1, 1000]
```

