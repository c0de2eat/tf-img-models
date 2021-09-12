# Results

## CIFAR10
For `CIFAR10`, a `ResNet18` with `small_input=True` is trained with `batch_size=128` for 50 epochs. Additional modules below are implemented on top of it.

### Backbones
| Backbone | Accuracy |
| :---: | :---: |
| ResNet18 | 93.03 |
| IBN | 93.66 |

### Attention Modules

#### Coarse
| Backbone | Accuracy |
| :---: | :---: |
| BAM | 92.87 |

### Fine
| Backbone | Accuracy |
| :---: | :---: |
| SE | 93.66 |
| CBAM | 93.61 |
