import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from os.path import join
from tqdm import tqdm
from datetime import datetime
from time import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# from keras.utils.vis_utils import plot_model

from network import get_model
from loss import AlternateLoss, MultiTaskLoss, AlphaLoss, AdaptiveTrimapLoss
from dataset import LiveComputedDataset
from utils import classic_grid, observer_grid, plot_to_image, generate_graph


#################
### VARIABLES ###
#################

img_size = (512, 512)
batch_size = 6
N_EPOCHS = 15000
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

        df = LiveComputedDataset("all_files", "/net/rnd/DEV/Datasets_DL/alpha_matting/", img_size=img_size, batch_size=batch_size)
        model, observers = get_model(img_size=img_size, depth=16)
        opt = Adam(learning_rate=0.001)
        
        loss_alpha_func = AlphaLoss()
        loss_trimap_func = AdaptiveTrimapLoss()
        loss_multitask_func = MultiTaskLoss()

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
        epoch = 0
        while True:
            epoch+=1
            set_profiler = True
            progress_bar = tqdm(df._df_train, desc=f"epoch={epoch}")
            progress_bar.set_postfix({"loss" : None})

            for x_batch, y_batch in progress_bar:
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch, training=True)
                    # Calcul des loss
                    loss_alpha = loss_alpha_func(y_batch, y_pred)
                    loss_trimap = loss_trimap_func(y_batch, y_pred)
                    loss = loss_multitask_func(y_batch, y_pred, loss_trimap, loss_alpha)
                    
                gradients = tape.gradient(loss, model.trainable_weights)
                opt.apply_gradients(zip(gradients, model.trainable_weights))

                s1 = tf.exp(0.5*model.layers[-1].kernel[0]).numpy()[0]
                s2 = tf.exp(0.5*model.layers[-1].kernel[1]).numpy()[0]

                loss_str = f"{loss.numpy():.4f}"                
                progress_bar.set_postfix({"loss" : loss_str})

                # Logging training data
                with train_writer.as_default():
                    tf.summary.scalar("AlphaLoss", loss_alpha, step=i)
                    tf.summary.scalar("TrimapLoss", loss_trimap, step=i)
                    tf.summary.scalar("MultiTaskLoss", loss, step=i)
                    tf.summary.scalar("S1", s1, step=i)
                    tf.summary.scalar("S2", s2, step=i)

                #  Logging testing and images
                if time() - last_test > PERIOD_TEST:
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
        
                    model.save_weights(join(save_dir, datetime.now().strftime("%m-%d_%Hh%M") + ".h5"), save_format="h5")
                    last_test = time()
                    # Loss = []
                    # for x_batch, y_batch in df._df_test:
                    #     y_pred = model(x_batch, training=False)
                    #     loss_alpha = loss_alpha_func(y_batch, y_pred)
                    #     loss_trimap = loss_trimap_func(y_batch, y_pred)
                    #     loss = loss_multitask_func(y_batch, y_pred, loss_trimap, loss_alpha)
                    #     Loss.append(loss)

                    # mean = lambda L : sum(L)/len(L) if len(L) > 0 else -1

                    with test_writer.as_default():
                        # tf.summary.scalar("MultiTaskLoss", mean(Loss), step=i)

                        fig_classic = classic_grid(df._df_train, df._n_val, model)
                        tf.summary.image("Validation Set", plot_to_image(fig_classic), step=img_index)

                        fig_observers = observer_grid(df._df_train, df._n_val, observers)
                        tf.summary.image("Observations", plot_to_image(fig_observers), step=img_index)

                        img_index+=1

                # Logging profiler info
                if epoch == 5 and set_profiler:
                    tf.profiler.experimental.start(join(log_dir, "profiler/"))
                    set_profiler = False
                if epoch == 10 and set_profiler:
                    tf.profiler.experimental.stop()
                    set_profiler = False
                
                # Next loop
                i+=1
        succeed = True
    except tf.errors.ResourceExhaustedError as e:
        batch_size = max(1, batch_size-1)
        print(f"Got OOM : reducing batch size to {batch_size}")