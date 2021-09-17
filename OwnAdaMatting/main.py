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

from keras.utils.vis_utils import plot_model

from network import get_model
from dataset import AdaMattingDataset
from utils import image_grid, plot_to_image, generate_graph


#################
### VARIABLES ###
#################

img_size = (512, 512)
batch_size = 12
N_EPOCHS = 150
PERIOD_TEST = 60*60 # Temps en seconde entre chaque test
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
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=BinaryCrossentropy(),
            metrics=["accuracy", "mse"]
        )

        ###################
        ### TENSORBOARD ###
        ###################
        train_writer = tf.summary.create_file_writer(join(log_dir, f"train/"))
        test_writer = tf.summary.create_file_writer(join(log_dir, f"test/"))
        graph_writer = tf.summary.create_file_writer(join(log_dir, f"graph//"))
        
        # Graph
        generate_graph(graph_writer, model)

        #####################
        ### TRAINING LOOP ###
        #####################
        i = 0
        img_index = 0
        for epoch in range(N_EPOCHS):
            model.save(join(save_dir, datetime.now().strftime("%m-%d_%Hh%M") + ".h5"))
            for x_batch, y_batch in tqdm(df._df_train, desc=f"epoch={epoch}"):
                # Training
                loss, acc, mse = model.train_on_batch(x_batch, y_batch)

                # Logging training data
                with train_writer.as_default():
                    tf.summary.scalar("Loss", loss, step=i)
                    tf.summary.scalar("Accuracy", acc, step=i)
                    tf.summary.scalar("MSE", mse, step=i)

                #  Logging testing and images
                if time() - last_test > PERIOD_TEST:
                    last_test = time()
                    Loss, Acc, Mse = [],[],[]
                    for x_batch, y_batch in df._df_test:
                        loss, acc, mse = model.test_on_batch(x_batch, y_batch)
                        Loss.append(loss)
                        Acc.append(acc)
                        Mse.append(mse)

                    mean = lambda L : sum(L)/len(L) if len(L) > 0 else -1

                    with test_writer.as_default():
                        tf.summary.scalar("Loss", mean(Loss), step=i)
                        tf.summary.scalar("Accuracy", mean(Acc), step=i)
                        tf.summary.scalar("MSE", mean(Mse), step=i)

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
        batch_size = max(1, int(batch_size*0.75))
        print(f"Got OOM : reducing batch size to {batch_size}")