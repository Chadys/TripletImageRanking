from pathlib import Path

import tensorflow as tf

import load_data

train_images = load_data.get_train_easy_images()

IMG_SIZE = 225
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 2
GAP_PARAMETER = 0.2
train_steps_per_epoch = len(train_images) // BATCH_SIZE

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# TODO NCHW


class MultiScaleNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__(name="multiscale_network")
        # self.convnet = tf.keras.applications.NASNetLarge(
        #     input_shape=IMG_SHAPE,
        #     include_top=False,
        #     weights="imagenet"
        # )
        self.convnet = tf.keras.applications.InceptionResNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights="imagenet"
        )
        # self.convnet = tf.keras.applications.Xception(
        #     input_shape=IMG_SHAPE,
        #     include_top=False,
        #     weights="imagenet"
        # )  # 'channel_last' only
        self.convnet.trainable = False

        # from paper : SubSample 4:1 and strides of 4 -> strides = 4*4=16
        self.conv_1 = tf.keras.layers.Conv2D(96, 8, 16, "same")
        self.maxpool_1 = tf.keras.layers.MaxPool2D(7, 4, "same")

        # from paper : SubSample 8:1 and strides of 4 -> strides = 8*4=32
        self.conv_2 = tf.keras.layers.Conv2D(96, 8, 32, "same")
        self.maxpool_2 = tf.keras.layers.MaxPool2D(3, 2, "same")

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(4096)

    def call(self, inputs, **kwargs):
        features_x = self.convnet(inputs)
        features_x = self.flatten(features_x)
        features_x = tf.math.l2_normalize(features_x, axis=-1)

        sub_x1 = self.conv_1(inputs)
        sub_x1 = self.maxpool_1(sub_x1)
        sub_x1 = self.flatten(sub_x1)

        sub_x2 = self.conv_2(inputs)
        sub_x2 = self.maxpool_2(sub_x2)
        sub_x2 = self.flatten(sub_x2)

        sub_x = tf.keras.layers.concatenate([sub_x1, sub_x2])
        sub_x = tf.math.l2_normalize(sub_x, axis=-1)

        x = tf.keras.layers.concatenate([features_x, sub_x])
        x = self.dense(x)
        x = tf.math.l2_normalize(x, axis=-1)
        return x


def triplet_loss(anchor, positive, negative):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
    # same as
    # pos_dist = tf.multiply(2., tf.subtract(1, tf.math.reduce_sum(tf.math.multiply(anchor, positive), -1)))
    # but first is faster
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), -1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), GAP_PARAMETER)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)


    return loss

# TODO add lambda regularization