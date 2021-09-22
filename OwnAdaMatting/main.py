import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from os.path import join
from tqdm import tqdm
from datetime import datetime
from time import time

import tensorflow as tf
from keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# from keras.utils.vis_utils import plot_model

from network import get_model, MultiTaskLoss
from dataset import AdaMattingDataset
from utils import image_grid, plot_to_image, generate_graph


#################
### VARIABLES ###
#################

img_size = (256, 256)
batch_size = 25
N_EPOCHS = 150
PERIOD_TEST = 60*1 # Temps en seconde entre chaque test
last_test = time()

###################
### PREPARATION ###
###################
succeed = False
while not succeed:
    try:
        date = datetime.now().strftime("%m-%d_%Hh%M")
        print(date)
        log_dir = f'OwnAdaMatting/logs/{date}/'
        save_dir = f'OwnAdaMatting/saves/{date}/'

        df = AdaMattingDataset("train", "/net/rnd/DEV/Datasets_DL/alpha_matting/", img_size=img_size, batch_size=batch_size)
        model = get_model(img_size=img_size, depth=16)
        loss_function = MultiTaskLoss()
        opt = Adam(learning_rate=0.001)
        # model.compile(
        #     optimizer=Adam(learning_rate=0.0001),
        #     loss=MultiTaskLoss(),
        # )

        ###################
        ### TENSORBOARD ###
        ###################
        train_writer = tf.summary.create_file_writer(join(log_dir, f"train/"))
        test_writer = tf.summary.create_file_writer(join(log_dir, f"test/"))
        
        # Graph
        generate_graph(test_writer, model)

        #####################
        ### TRAINING LOOP ###
        #####################
        i = 0
        img_index = 0
        for epoch in range(N_EPOCHS):
            progress_bar = tqdm(df._df_train, desc=f"epoch={epoch}")
            progress_bar.set_postfix({"loss" : None})
            for x_batch, y_batch in progress_bar:
                # Training
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch, training=True)
                    loss = loss_function(y_batch, y_pred)

                gradients = tape.gradient(loss, model.trainable_weights)
                opt.apply_gradients(zip(gradients, model.trainable_weights))
                s1 = f"{tf.exp(0.5*model.layers[-1].kernel[0]).numpy()[0]:.4f}"
                s2 = f"{tf.exp(0.5*model.layers[-1].kernel[1]).numpy()[0]:.4f}"
                loss_str = f"{loss.numpy():.4f}"
                progress_bar.set_postfix({
                    "loss" : loss_str,
                    "s1" : s1,
                    "s2" : s2})

                # Logging training data
                with train_writer.as_default():
                    tf.summary.scalar("Loss", loss, step=i)

                #  Logging testing and images
                if time() - last_test > PERIOD_TEST:
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
        
                    model.save_weights(join(save_dir, datetime.now().strftime("%m-%d_%Hh%M") + ".h5"), save_format="h5")
                    last_test = time()
                    Loss, Acc, Mse = [],[],[]
                    for x_batch, y_batch in df._df_test:
                        y_pred = model(x_batch, training=False)
                        loss = loss_function(y_batch, y_pred)
                        Loss.append(loss)

                    mean = lambda L : sum(L)/len(L) if len(L) > 0 else -1

                    with test_writer.as_default():
                        tf.summary.scalar("Loss", mean(Loss), step=i)

                        fig = image_grid(df._df_val, model, df._n_val)
                        tf.summary.image("Validation Set", plot_to_image(fig), step=img_index)
                        img_index+=1

                # Logging profiler info
                if i == 10:
                    tf.profiler.experimental.start(join(log_dir, "profiler/"))
                if i == 20:
                    tf.profiler.experimental.stop()
                
                # Next loop
                i+=1
        succeed = True
    except tf.errors.ResourceExhaustedError as e:
        batch_size = max(1, batch_size-1)
        print(f"Got OOM : reducing batch size to {batch_size}")