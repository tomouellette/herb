# :herb: herb

A rough collection of personal scripts, utilities, and implementations for experimenting, building, and training neural networks. Tired of abstraction bloat? Backbones and models are all self-contained in single files.

## Setup

Only `torch`, `torchvision`, and `huggingface` (accelerate) are required for running models and scripts. Additional dependencies are used for optional tests (see `requirements.txt`).

```bash
python3 -m pip install -r requirements.txt
```

By default, all I/O for various models are handled through the basic pytorch `ImageFolder` dataset. If you want faster I/O that works across shards, you can replace these blocks with a `webdataset` dataset.

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

## Models

### DINO

```python3
# Run test training on MNIST (~96.5% linear probe accuracy)
python3 -m models.dino --test True --epochs 10

# Run test training on MNIST and generate a GIF of PCA'd embeddings
GIF=1 python3 -m models.dino --test True --epochs 10

# Run standard training
python3 -m models.dino \
    --image_folder 'data' \
    --image_size 28 \
    --backbone 'mlp_mixer_small' \
    --channels 1 \
    --epochs 10 \
    --batch_size 256 \
    --max_lr 1e-4 \
    --min_lr 1e-6 \
    --lr_warmup_fraction 0.1 \
    --projector_hidden_dim 256 \
    --projector_k 256 \
    --projector_layers 4 \
    --projector_batch_norm False \
    --projector_l2_norm True \
    --momentum_center 0.9 \
    --momentum_teacher 0.996 \
    --global_crops_scale 0.5 1.0 \
    --local_crops_scale 0.3 0.7 \
    --n_augments 4 \
    --t_teacher_start 0.04 \
    --t_teacher_end 0.02 \
    --t_teacher_warmup_fraction 0.1 \
    --t_student 0.1 \
    --silent False
```
