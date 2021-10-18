import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from os.path import join
from tqdm import tqdm
from datetime import datetime
from time import time

import tensorflow as tf
# tf.data.experimental.enable_debug_mode()
from tensorflow.keras.optimizers import Adam

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# from keras.utils.vis_utils import plot_model

from network import get_model
from loss import MultiTaskLoss, AlphaLoss, AdaptiveTrimapLoss
from dataset import LiveComputedDataset, DeepDataset
from utils import classic_grid, observer_grid, plot_to_image, generate_graph

mean = lambda L : sum(L)/len(L) if len(L) > 0 else -1


#################
### VARIABLES ###
#################

size = 10
img_size = (size*32, size*32)
batch_size = 10
PERIOD_TEST = 60*7 # Temps en seconde entre chaque test
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
 
        # df = LiveComputedDataset("all_files", "/net/rnd/DEV/Datasets_DL/alpha_matting/", img_size=img_size, batch_size=batch_size)
        df = DeepDataset("/net/rnd/DEV/Datasets_DL/alpha_matting/deep38/", batch_size=batch_size, img_size=img_size, size_dividor=32, max_size_factor=3)
        _ , _, model, observers = get_model(depth=32)
        # model.load_weights("/net/homes/r/rseailles/Deep/OwnAdaMatting/saves/10-18_12h17/10-18_12h32.h5")
        opt = Adam(learning_rate=0.0001)
        
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
        test_index = 0
        epoch = 0
        while True:
            epoch+=1
            progress_bar = tqdm(df._ds_train, desc=f"epoch={epoch}")

            for x_batch, y_batch in progress_bar:
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch, training=True)
                    loss_trimap = loss_trimap_func(y_batch, y_pred)
                    
                gradients = tape.gradient(loss_trimap, model.trainable_weights)
                opt.apply_gradients(zip(gradients, model.trainable_weights))

                # Logging training data
                with train_writer.as_default():
                    tf.summary.scalar("TrimapLoss", loss_trimap, step=i)
                    i+=1

                #  Logging testing and images
                if time() - last_test > PERIOD_TEST:
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
        
                    model.save_weights(join(save_dir, datetime.now().strftime("%m-%d_%Hh%M") + ".h5"), save_format="h5")
                    last_test = time()
                    
                    Loss_trimap= []
                    for x_batch, y_batch in tqdm(df._ds_test, desc="TEST"):
                        y_pred = model(x_batch, training=True)
                        loss_trimap = loss_trimap_func(y_batch, y_pred)
                        Loss_trimap.append(loss_trimap.numpy())

                    with test_writer.as_default():
                        tf.summary.scalar("TrimapLoss", mean(Loss_trimap), step=i)

                        fig_classic = classic_grid(df._ds_test, df._n_images, model)
                        tf.summary.image("Test Set", plot_to_image(fig_classic), step=test_index)

                    test_index+=1

            # Logging profiler info
            # if epoch == 1:
            #     tf.profiler.experimental.start(join(log_dir, "profiler/"))
            # if epoch == 2:
            #     tf.profiler.experimental.stop()

    except tf.errors.ResourceExhaustedError as e:
        batch_size = max(1, batch_size-1)
        print(f"Got OOM : reducing batch size to {batch_size}")