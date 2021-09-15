import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras.losses import BinaryCrossentropy

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# from keras.utils.vis_utils import plot_model

from network import get_model
from dataset import AdaMattingDataset

img_size = (256, 256)
df = AdaMattingDataset("train", "/net/rnd/DEV/Datasets_DL/alpha_matting/", img_size=img_size, batch_size=32)
model = get_model(img_size=img_size, depth=16)
# model.summary()
model.compile(
    optimizer="adam",
    loss=BinaryCrossentropy(),
    metrics=["accuracy", "mse"]
)
# plot_model(model, show_shapes=True)

log_dir = "OwnAdaMatting/logs/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    df._df_train, 
    validation_data=df._df_test, 
    epochs=10, 
    callbacks=[tensorboard_callback],
)