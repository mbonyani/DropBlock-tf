# DropBlock-tf
This is a TensorFlow implementation of the following paper: DropBlock: A regularization method for convolutional networks
>arXiv. https://arxiv.org/abs/1810.12890

```python
import numpy as np
import tensorflow as tf
from utils.dropblock import DropBlock2D

# only support `channels_last` data format
net = tf.keras.layers.Conv2D(256, (3, 3), padding='same',kernel_initializer=initializer)(net)
net = tf.keras.layers.BatchNormalization()(net)
net = tf.keras.layers.LeakyReLU(alpha=0.1)(net)
net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)
net = DropBlock2D(keep_prob=0.5, block_size=3)(net,training=True)

```
