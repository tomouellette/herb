# :herb: herb

A rough collection of personal scripts, utilities, and implementations for experimenting, building, and training neural networks. Tired of abstraction bloat? Backbones and models are all self-contained in single files.

## Setup

Only `torch`, `torchvision`, and `huggingface` (accelerate) are required for running models and scripts. Additional dependencies are used for optional tests (see `requirements.txt`).

```bash
python3 -m pip install -r requirements.txt
```

## Models

### DINO

```python3
# Run test training (~97% linear probe accuracy on MNIST)
GIF=1 python3 -m models.dino --test True

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
