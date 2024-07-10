# :herb: herb

A rough collection of personal scripts, utilities, and implementations for experimenting, building, and training neural networks. Most things setup for toying around in the single CPU/GPU setting.

### Examples

Some toy examples to build from when starting new projects.

#### DINO

```python3
# Self-supervised training of a simple MLP to >90% linear probe accuracy on MNIST with DINO
python3 -m examples.train_dino \
    --image_size 28 \
    --channels 1 \
    --projector_hidden_dim 256 \
    --projector_k 4096 \
    --projector_layers 4 \
    --projector_batch_norm True \
    --projector_l2_norm True \
    --t_teacher 0.04 \
    --t_student 0.9 \
    --crop_local_scales 0.9 1.0 \
    --crop_global_scales 0.9 1.0 \
    --ema_decay_teacher 0.99 \
    --ema_decay_center 0.9 \
    --batch_size 64 \
    --epochs 11 \
    --lr_max 0.0005 \
    --lr_min 0.000001 \
    --lr_warmup 1 \
    --weight_decay_start 0.04 \
    --weight_decay_end 0.4 \
    --clip_grad 0. \
    --freeze_projector 1 \
    --anneal_momentum True \
    --seed 123456 \
    --t_teacher_final 0.04 \
    --t_teacher_warmup 0.02 \
    --t_teacher_warmup_epochs 10 \
    --device mps
```
