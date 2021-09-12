# TF Image Models
TensorFlow image models for image related tasks. Please note that the implementation is based on my best understanding of the paper if official implementation is not available and there is no guarantee that it is correct.

Folder structure:
- `docker` contains the `Dockerfile` mainly for development purposes.
- `tfim` contains all the implementtions.
- `scripts` contains the training script to run experiments on.

---

## Modeling
[ResNet](https://arxiv.org/abs/1512.03385) variants are used and all layers and modules are implmented on top of it..

### Variants
- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)

### Layers
- [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net](https://arxiv.org/abs/1807.09441)

### Modules
- Attentions
    - [Squeeze-and-Excitation (SE) Networks](https://arxiv.org/abs/1709.01507)
    - [Bottleneck Attention Module (BAM)](https://arxiv.org/abs/1807.06514)
    - [Convolutional Block Attention Module (CBAM)](https://arxiv.org/abs/1807.06521)


<!-- ### Architectures
- [Feature Pyramid Networks (FPN) for Object Detection](https://arxiv.org/abs/1612.03144)
    - ResNet -->

---

## Install
Run the following command to install `tfim` package in the current Python environment.

```bash
pip install git+https://github.com/c0de2eat/tf-img-models
```
