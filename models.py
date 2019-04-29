import tensorflow as tf


class MultiScaleNetwork(tf.keras.Model):
    def __init__(self, img_shape, embeddings_dim):
        super().__init__(name="multiscale_network")
        # self.convnet = tf.keras.applications.NASNetLarge(
        #     input_shape=IMG_SHAPE,
        #     include_top=False,
        #     weights="imagenet"
        # )
        self.convnet = tf.keras.applications.InceptionResNetV2(
            input_shape=img_shape,
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
        self.dense = tf.keras.layers.Dense(embeddings_dim)

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


class DeepRankingNetwork(tf.keras.Model):
    def __init__(self, img_shape, embeddings_dim):
        super().__init__(name="deepranking_network")
        self.multiscale = MultiScaleNetwork(img_shape=img_shape, embeddings_dim=embeddings_dim)

    def call(self, inputs, **kwargs):
        return tf.keras.layers.concatenate([self.multiscale(x) for x in inputs])