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
from dataset import DeepDataset
from utils import classic_grid, observer_grid, plot_to_image, generate_graph

mean = lambda L : sum(L)/len(L) if len(L) > 0 else -1


#################
### VARIABLES ###
#################

size = 7
img_size = size*32
batch_size = 7
PERIOD_TEST = 60*15 # Temps en seconde entre chaque test
last_test = time()

###################
### PREPARATION ###
###################

succeed = False
while not succeed:
    try:
        ds = DeepDataset("/net/rnd/DEV/Datasets_DL/alpha_matting/deep38/", batch_size=batch_size, squared_img_size=img_size, max_size_factor=3.0)
        m_training , _, m_trimap, observers = get_model(depth=32)
        m_trimap.load_weights("/net/homes/r/rseailles/Deep/OwnAdaMatting/saves/10-19_19h30/10-20_18h13.h5")
        m_trimap.trainable = False
        m_training.load_weights("/net/homes/r/rseailles/Deep/OwnAdaMatting/saves/10-19_13h41/10-20_16h39.h5")

        opt = Adam(learning_rate=0.0001)
        
        loss_alpha_func = AlphaLoss()
        loss_trimap_func = AdaptiveTrimapLoss()
        loss_multitask_func = MultiTaskLoss()

        ###################
        ### TENSORBOARD ###
        ###################
        date = datetime.now().strftime("%m-%d_%Hh%M")
        print(date)
        log_dir = f'OwnAdaMatting/logs/{date}/'
        save_dir = f'OwnAdaMatting/saves/{date}/'
        train_writer = tf.summary.create_file_writer(join(log_dir, f"train/"))
        test_writer = tf.summary.create_file_writer(join(log_dir, f"test/"))
        
        # Graph
        generate_graph(test_writer, m_training)

        #####################
        ### TRAINING LOOP ###
        #####################
        i = 0
        test_index = 0
        epoch = 0
        while True:
            epoch+=1
            progress_bar = tqdm(ds.ds_train, desc=f"epoch={epoch}")

            for x_batch, y_batch in progress_bar:
                with tf.GradientTape() as tape:
                    y_pred = m_training(x_batch, training=True)
                    # Calcul des loss
                    loss_alpha = loss_alpha_func(y_batch, y_pred)
                    loss_trimap = loss_trimap_func(y_batch, y_pred)
                    loss = loss_multitask_func(y_batch, y_pred, loss_trimap, loss_alpha)
                    
                gradients = tape.gradient(loss, m_training.trainable_weights)
                opt.apply_gradients(zip(gradients, m_training.trainable_weights))

                s1 = tf.exp(0.5*m_training.layers[-1].kernel[0]).numpy()[0]
                s2 = tf.exp(0.5*m_training.layers[-1].kernel[1]).numpy()[0]

                # Logging training data
                with train_writer.as_default():
                    tf.summary.scalar("AlphaLoss", loss_alpha, step=i)
                    tf.summary.scalar("TrimapLoss", loss_trimap, step=i)
                    tf.summary.scalar("MultiTaskLoss", loss, step=i)
                    tf.summary.scalar("S1", s1, step=i)
                    tf.summary.scalar("S2", s2, step=i)
                    i+=1

                #  Logging testing and images
                if time() - last_test > PERIOD_TEST:

                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
        
                    m_training.save_weights(join(save_dir, datetime.now().strftime("%m-%d_%Hh%M") + ".h5"), save_format="h5")
                    
                    Loss_alpha, Loss_trimap, Loss = [],[],[]
                    for x_batch, y_batch in tqdm(ds.ds_test, desc="TEST"):
                        y_pred = m_training(x_batch, training=True)
                        loss_alpha = loss_alpha_func(y_batch, y_pred)
                        loss_trimap = loss_trimap_func(y_batch, y_pred)
                        loss = loss_multitask_func(y_batch, y_pred, loss_trimap, loss_alpha)

                        Loss_alpha.append(loss_alpha.numpy())
                        Loss_trimap.append(loss_trimap.numpy())
                        Loss.append(loss.numpy())

                    with test_writer.as_default():
                        tf.summary.scalar("AlphaLoss", mean(Loss_alpha), step=i)
                        tf.summary.scalar("TrimapLoss", mean(Loss_trimap), step=i)
                        tf.summary.scalar("MultiTaskLoss", mean(Loss), step=i)

                        fig_classic = classic_grid(ds.ds_test, 5, m_training)
                        tf.summary.image("Test Set", plot_to_image(fig_classic), step=test_index)
                        fig_observers = observer_grid(ds.ds_test, 5, observers)
                        tf.summary.image("Observations", plot_to_image(fig_observers), step=test_index)

                    test_index+=1
                    last_test = time()

            # Logging profiler info
            # if epoch == 1:
            #     tf.profiler.experimental.start(join(log_dir, "profiler/"))
            # if epoch == 2:
            #     tf.profiler.experimental.stop()

    except tf.errors.ResourceExhaustedError as e:
        batch_size = max(1, batch_size-1)
        print(f"Got OOM : reducing batch size to {batch_size}")
