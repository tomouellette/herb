# :herb: herb

A rough collection of personal scripts, utilities, and implementations for experimenting, building, and training neural networks. Most things setup for toying around in the single CPU/GPU setting.

### Examples

Some toy examples to build from when starting new projects.

#### DINO

```python3
# Self-supervised training of a simple MLP to >90% linear probe accuracy on MNIST with DINO
python3 -m examples.train_dino \
    --device mps \
    --epochs 64 \ # 10 epochs should be enough; 64 is purely for making a cool GIF
    --batch_size 16 \
    --t_teacher 0.01 \
    --ema_decay_teacher 0.998 \
    --crop_local_scales 0.7 1.0 \
    --projector_batch_norm True \
    --freeze_projector 1 \
    --lr_max 0.0001 \
    --lr_min 0.000001 \
    --lr_warmup 10 \
    --anneal_momentum False
```
# aloe
