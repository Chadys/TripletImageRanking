import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, gap_parameter, embeddings_dim, batch_size, reduction=losses_utils.ReductionV2.SUM, name="triplet_loss"):
        self.gap_parameter = gap_parameter
        self.embeddings_dim = embeddings_dim
        self.batch_size = batch_size
        super().__init__(reduction=reduction, name=name)

    def call(self, _, triplet):
        anchor, positive, negative = (
            triplet[:, i: i + self.embeddings_dim]
            for i in range(0, self.embeddings_dim * 3, self.embeddings_dim)
        )
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), -1)
        # because vectors are l2 normed, same as
        # pos_dist = tf.multiply(2., tf.subtract(1, tf.math.reduce_sum(tf.math.multiply(anchor, positive), -1)))
        # but first is faster

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.gap_parameter)
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0), -1)  * (1. / self.batch_size) # because of parallel strategy

        return loss
