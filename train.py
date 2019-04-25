from pathlib import Path

import tensorflow as tf

import load_data

train_images = load_data.get_train_easy_images()

BATCH_SIZE = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 2
train_steps_per_epoch = len(train_images) // BATCH_SIZE

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


class MultiScaleNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__(name="multiscale_network")
        # TODO convnet
        # from paper : SubSample 4:1 and strides of 4 -> strides = 4*4=16
        self.conv_1 = tf.keras.layers.Conv2D(96, 8, 16, "same")
        self.maxpool_1 = tf.keras.layers.MaxPool2D(7, 4, "same")

        # from paper : SubSample 8:1 and strides of 4 -> strides = 8*4=32
        self.conv_2 = tf.keras.layers.Conv2D(96, 8, 32, "same")
        self.maxpool_2 = tf.keras.layers.MaxPool2D(3, 2, "same")

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(4096)

    def call(self, inputs, **kwargs):
        # TODO x -> convnet
        sub_x1 = self.conv_1(inputs)
        sub_x1 = self.maxpool_1(sub_x1)
        sub_x1 = self.flatten(sub_x1)

        sub_x2 = self.conv_2(inputs)
        sub_x2 = self.maxpool_2(sub_x2)
        sub_x2 = self.flatten(sub_x2)

        sub_x = tf.keras.layers.concatenate([sub_x1, sub_x2])
        sub_x = tf.math.l2_normalize(sub_x, axis=1)

        # TODO concat w/ convnet output then use dense and l2_norm
