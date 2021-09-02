# tf_img_models
TensorFlow common codes for computer vision related tasks.
TensorFlow utility library with implementation of models for image related tasks. Please note that the implementation is based on my best understanding and there is no guarantee that it is absolutely correct.

---

## Modeling
Models and layers that are currently implemented.

### Modules
- Residual modules from ResNet
- Attentions
    - [Bottleneck Attention Module (BAM)](https://arxiv.org/abs/1807.06514)
    - [Convolutional Block Attention Module (CBAM)](https://arxiv.org/abs/1807.06521)

### Backbones
- [ResNet](https://arxiv.org/abs/1512.03385)
    - [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579)
    - [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)
    - [Pre-activation: Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
    - External modules
        - BAM
        - CBAM
        - BAM + CBAM

<!-- ### Architectures
- [Feature Pyramid Networks (FPN) for Object Detection](https://arxiv.org/abs/1612.03144)
    - ResNet -->

---

## Others
### Training
- Cosine decay scheduler with warmup

### Data
- Common data related functions such as batching the dataset and etc.

---

## Build
### Docker image
Build the docker image for `tf_img_models`.

```bash
docker build -t tf_img_models -f docker/Dockerfile  .
```

### Python
Run the following command to install `tf_img_models` package in the current Python environment.

```bash
pip install -e .
```
