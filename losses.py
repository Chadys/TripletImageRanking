import tensorflow as tf


class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, gap_parameter, embeddings_dim, **kwargs):
        self.gap_parameter = gap_parameter
        self.embeddings_dim = embeddings_dim
        super().__init__(**kwargs, name="triplet_loss")

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
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), -1)

        return loss
