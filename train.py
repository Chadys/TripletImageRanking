# import os
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # waiting for integration from tf-nvidia to tf
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
# TODO add activation layers ?

print(f"image data format is {tf.keras.backend.image_data_format()}")
datasets = data.get_hard_images()
model = DeepRankingNetwork(IMG_SHAPE, EMBEDDINGS_DIM)
base_lr = 0.001
max_lr = 0.06
model.compile(
    optimizer=tf.keras.optimizers.Adam(base_lr),  # TODO choose optimizer
    loss=TripletLoss(GAP_PARAMETER, EMBEDDINGS_DIM, BATCH_SIZE),
    metrics=[],
)

steps_per_epoch = datasets["train"][0] // BATCH_SIZE
model.fit(
    datasets["train"][1],
    validation_data=datasets["validation"][1],
    validation_steps=datasets["validation"][0] // BATCH_SIZE,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch_index: base_lr + (max_lr-base_lr)*(max(0, (1-epoch_index))*1/(2.**(epoch_index-1)))),  # TODO test w/ ReduceLROnPlateau instead
        tf.keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath="saves/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor="val_loss", save_best_only=True, save_freq="epoch"),
        tf.keras.callbacks.TensorBoard(histogram_freq=5, batch_size=BATCH_SIZE, write_grads=True, write_images=True)
    ]
)