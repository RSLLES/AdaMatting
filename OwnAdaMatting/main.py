import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

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
from utils import classic_grid, observer_grid, val_grid, plot_to_image, generate_graph

mean = lambda L : sum(L)/len(L) if len(L) > 0 else -1


#################
### VARIABLES ###
#################

img_size = (512, 512)
batch_size = 8
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
 
        df = LiveComputedDataset("picky2", "/net/rnd/DEV/Datasets_DL/alpha_matting/", img_size=img_size, batch_size=batch_size)
        model, observers = get_model(img_size=img_size, depth=16)
        # model.load_weights("/net/homes/r/rseailles/Deep/OwnAdaMatting/saves/10-01_19h18/10-04_10h08.h5")
        opt = Adam(learning_rate=0.001)
        
        loss_alpha_func = AlphaLoss()
        loss_trimap_func = AdaptiveTrimapLoss()
        loss_multitask_func = MultiTaskLoss()

        ###################
        ### TENSORBOARD ###
        ###################
        train_writer = tf.summary.create_file_writer(join(log_dir, f"train/"))
        test_writer = tf.summary.create_file_writer(join(log_dir, f"test/"))
        val_writer = tf.summary.create_file_writer(join(log_dir, f"val/"))
        
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

            Loss_alpha, Loss_trimap, Loss, S1, S2 = [],[],[],[],[]
            for x_batch, y_batch in progress_bar:
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch, training=True)
                    # Calcul des loss
                    loss_alpha = loss_alpha_func(y_batch, y_pred)
                    loss_trimap = loss_trimap_func(y_batch, y_pred)
                    loss = loss_multitask_func(y_batch, y_pred, loss_trimap, loss_alpha)
                    
                gradients = tape.gradient(loss, model.trainable_weights)
                opt.apply_gradients(zip(gradients, model.trainable_weights))

                Loss_alpha.append(loss_alpha.numpy())
                Loss_trimap.append(loss_trimap.numpy())
                Loss.append(loss.numpy())
                S1.append(tf.exp(0.5*model.layers[-1].kernel[0]).numpy()[0])
                S2.append(tf.exp(0.5*model.layers[-1].kernel[1]).numpy()[0])

            # Logging training data
            with train_writer.as_default():
                tf.summary.scalar("AlphaLoss", mean(Loss_alpha), step=epoch)
                tf.summary.scalar("TrimapLoss", mean(Loss_trimap), step=epoch)
                tf.summary.scalar("MultiTaskLoss", mean(Loss), step=epoch)
                tf.summary.scalar("S1", mean(S1), step=epoch)
                tf.summary.scalar("S2", mean(S2), step=epoch)

            #  Logging testing and images
            if time() - last_test > PERIOD_TEST:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
    
                model.save_weights(join(save_dir, datetime.now().strftime("%m-%d_%Hh%M") + ".h5"), save_format="h5")
                last_test = time()
                
                Loss_alpha, Loss_trimap, Loss = [],[],[]
                for x_batch, y_batch in df._ds_test:
                    y_pred = model(x_batch, training=True)
                    loss_alpha = loss_alpha_func(y_batch, y_pred)
                    loss_trimap = loss_trimap_func(y_batch, y_pred)
                    loss = loss_multitask_func(y_batch, y_pred, loss_trimap, loss_alpha)

                    Loss_alpha.append(loss_alpha.numpy())
                    Loss_trimap.append(loss_trimap.numpy())
                    Loss.append(loss.numpy())

                with test_writer.as_default():
                    tf.summary.scalar("AlphaLoss", mean(Loss_alpha), step=epoch)
                    tf.summary.scalar("TrimapLoss", mean(Loss_trimap), step=epoch)
                    tf.summary.scalar("MultiTaskLoss", mean(Loss), step=epoch)

                    fig_classic = classic_grid(df._ds_test, df._n_images, model)
                    tf.summary.image("Test Set", plot_to_image(fig_classic), step=test_index)
                    fig_observers = observer_grid(df._ds_test, df._n_images, observers)
                    tf.summary.image("Observations", plot_to_image(fig_observers), step=test_index)
                
                # with val_writer.as_default():
                #     fig_val = val_grid(df._ds_val, 8, model)
                #     tf.summary.image("Validation Set", plot_to_image(fig_val), step=test_index)

                test_index+=1

            # Logging profiler info
            if epoch == 5:
                tf.profiler.experimental.start(join(log_dir, "profiler/"))
            if epoch == 10:
                tf.profiler.experimental.stop()
        
        succeed = True
    except tf.errors.ResourceExhaustedError as e:
        batch_size = max(1, batch_size-1)
        print(f"Got OOM : reducing batch size to {batch_size}")