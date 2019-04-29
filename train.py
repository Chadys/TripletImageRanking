import tensorflow as tf

import data

train_images = data.get_hard_images()

IMG_SIZE = data.IMG_SIZE
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 2
GAP_PARAMETER = 0.2
train_steps_per_epoch = len(train_images) // BATCH_SIZE

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# TODO NCHW


def triplet_loss(anchor, positive, negative):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), -1)
    # because vectors are l2 normed, same as
    # pos_dist = tf.multiply(2., tf.subtract(1, tf.math.reduce_sum(tf.math.multiply(anchor, positive), -1)))
    # but first is faster

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), GAP_PARAMETER)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


# TODO add lambda regularization