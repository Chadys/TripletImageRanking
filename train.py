from pathlib import Path

import tensorflow as tf

import load_data

train_images = load_data.train_easy_images()

BUFFER_SIZE = 400
BATCH_SIZE = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE
LAMBDA = 100
EPOCHS = 2
train_steps_per_epoch = len(train_images) // BATCH_SIZE

strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# use L2-norm