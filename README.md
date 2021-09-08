# TF Image Models
TensorFlow common codes for computer vision related tasks.
TensorFlow utility library with implementation of models for image related tasks. Please note that the implementation is based on my best understanding if official is not available and there is no guarantee that it is absolutely correct.

Folder structure:
- `docker` contains the `Dockerfile` mainly for development purposes.
- `tfim` contains all the implemented codes.
- `scripts` contains the script to test the implementation on CIFAR10.

---

## Modeling
Models and layers that are currently implemented.

### Modules
- Attentions
    - [Bottleneck Attention Module (BAM)](https://arxiv.org/abs/1807.06514)
    - [Convolutional Block Attention Module (CBAM)](https://arxiv.org/abs/1807.06521)

### Backbones
- [ResNet](https://arxiv.org/abs/1512.03385)
    <!-- - [Pre-activation: Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) -->

<!-- ### Architectures
- [Feature Pyramid Networks (FPN) for Object Detection](https://arxiv.org/abs/1612.03144)
    - ResNet -->

---

## Install
Run the following command to install `tfim` package in the current Python environment.

```bash
pip install git+https://github.com/c0de2eat/tf-img-models
```
