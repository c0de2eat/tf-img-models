# TF Image Models

TensorFlow image models for image related tasks. Please note that the implementation is based on my best understanding of the paper if official implementation is not available and there is no guarantee that it is correct.

Folder structure:

- `tfim` contains all the implementtions.
- `scripts` contains the training script to run on CIFAR10.

---

## Modeling

### ResNet

ResNet variants.

- [ResNet](https://arxiv.org/abs/1512.03385) - `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`,
- [ResNeXt](https://arxiv.org/abs/1611.05431) - `resnext50_32x4d`

### Layers

- [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net](https://arxiv.org/abs/1807.09441)

<!-- ### Modules

- Attentions
  - [Squeeze-and-Excitation (SE) Networks](https://arxiv.org/abs/1709.01507)
  - [Bottleneck Attention Module (BAM)](https://arxiv.org/abs/1807.06514)
  - [Convolutional Block Attention Module (CBAM)](https://arxiv.org/abs/1807.06521) -->

<!-- ### Architectures
- [Feature Pyramid Networks (FPN) for Object Detection](https://arxiv.org/abs/1612.03144)
    - ResNet -->

---

## Install

Run the following command to install `tfim` package in the current Python environment.

```bash
pip install git+https://github.com/c0de2eat/tf-img-models
```
