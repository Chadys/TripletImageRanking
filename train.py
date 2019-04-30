import tensorflow as tf

tf.keras.backend.set_image_data_format(
    "channels_last"
)  # TODO change to channels_first when on GPU

import data
from losses import TripletLoss
from models import DeepRankingNetwork

IMG_SIZE = data.IMG_SIZE
IMG_SHAPE = (
    (IMG_SIZE, IMG_SIZE, 3)
    if tf.keras.backend.image_data_format() == "channels_last"
    else (3, IMG_SIZE, IMG_SIZE)
)
EPOCHS = 2
GAP_PARAMETER = 0.2
EMBEDDINGS_DIM = 4096
BATCH_SIZE = data.BATCH_SIZE


# strategy = tf.distribute.MirroredStrategy()
# print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# TODO add lambda regularization
# TODO other regularizer (batch norm, dropout etc)
# TODO CLR

print(f"image data format is {tf.keras.backend.image_data_format()}")
datasets = data.get_hard_images()
model = DeepRankingNetwork(IMG_SHAPE, EMBEDDINGS_DIM)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=TripletLoss(GAP_PARAMETER, EMBEDDINGS_DIM),
    metrics=["accuracy"],
)

model.fit(
    datasets["train"][1],
    validation_data=datasets["validation"][1],
    validation_steps=datasets["validation"][0] // BATCH_SIZE,
    steps_per_epoch=datasets["train"][0] // BATCH_SIZE,
    epochs=EPOCHS,
)
