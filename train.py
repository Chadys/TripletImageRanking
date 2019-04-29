import tensorflow as tf

import data
from models import DeepRankingNetwork

IMG_SIZE = data.IMG_SIZE
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
EPOCHS = 2
GAP_PARAMETER = 0.2
EMBEDDINGS_DIM = 4096
BATCH_SIZE = data.BATCH_SIZE


def make_triplet_loss(gap_parameter, embeddings_dim):
    def triplet_loss(_, triplet):
        anchor, positive, negative = (
            triplet[:, i : i + embeddings_dim]
            for i in range(0, embeddings_dim * 3, embeddings_dim)
        )
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), -1)
        # because vectors are l2 normed, same as
        # pos_dist = tf.multiply(2., tf.subtract(1, tf.math.reduce_sum(tf.math.multiply(anchor, positive), -1)))
        # but first is faster

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), gap_parameter)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

        return loss

    return triplet_loss


# strategy = tf.distribute.MirroredStrategy()
# print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# TODO NCHW
# TODO add lambda regularization


datasets = data.get_hard_images()
model = DeepRankingNetwork(IMG_SHAPE, EMBEDDINGS_DIM)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=make_triplet_loss(GAP_PARAMETER, EMBEDDINGS_DIM),
    metrics=["accuracy"],
)

model.fit(
    datasets["train"][1],
    validation_data=datasets["validation"][1],
    validation_steps=datasets["validation"][0] // BATCH_SIZE,
    steps_per_epoch=datasets["train"][0] // BATCH_SIZE,
    epochs=EPOCHS,
)
