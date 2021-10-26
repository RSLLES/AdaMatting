import os
import io
import stat
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from datetime import datetime

######################
### Classe de base ###
######################

class Engine:
    ############
    ### Init ###
    ############

    def __init__(self, dataset) -> None:
        self.ds = dataset
        self.image_index = 0
        self.epoch = 0
        self.i = 0
        self.reset_train_metrics()
        self.reset_test_metrics()
        self.reset_saving_condition()
        self.reset_testing_condition()
        self.reset_testing_log_condition()
        self.reset_training_log_condition()
    
    ##############################
    ### Train and test methods ###
    ##############################

    def train(self):
        self.check_if_ready_for_training()

        # Boucle
        while self.main_loop_condition():
            self.epoch += 1
            progress_bar = tqdm(self.ds.ds_train, desc=f"epoch={self.epoch}")

            # Boucle de Train
            for x_batch, y_batch in progress_bar:

                ### Train ###
                self.train_one_batch(x_batch, y_batch)

                ### Logging training ###
                if self.training_log_condition():
                    self.log_train()
                    self.reset_training_log_condition()

                ### Saving weights ###
                if self.saving_condition():
                    self.save()
                    self.reset_saving_condition()

                ### Test ###
                if self.testing_condition():
                    self.test()
                    if self.testing_log_condition():
                        self.log_test()
                        self.log_images_test()
                        self.reset_testing_log_condition()
                    self.reset_testing_condition()

                self.i +=1
                    


    def train_one_batch(self, x_train, y_train):
        with tf.GradientTape() as tape:
            y_pred = self.model(x_train, training=True)
            losses = self.compute_losses(y_train, y_pred)
            self.update_train_metrics(losses)
            loss = self.extract_training_loss(losses)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))


    def test(self):
        for x_test, y_test in tqdm(self.ds.ds_test, desc="TEST"):
            y_pred = self.model(x_test, training=False)
            losses = self.compute_losses(y_test, y_pred)
            self.update_test_metrics(losses)


    #########################
    ### Utility functions ###
    #########################

    def check_if_ready_for_training(self):
        list_mandatory_attributes = [
            "ds",
            "model",
            "optimizer",
            "train_writer",
            "test_writer",
            "save_dir",
            "log_dir"
        ]
        list_undefined_attributes = [
            attr for attr in list_mandatory_attributes if not hasattr(self, attr)
        ]
        if len(list_undefined_attributes) == 0:
            return True        
        raise ValueError(f"Missing some attributes : {list_undefined_attributes}")


    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.model.save_weights(
            os.path.join(
                self.save_dir, 
                datetime.now().strftime("%m-%d_%Hh%M") + ".h5"
                ), save_format="h5"
            )

    def load(self, path_to_weights_folder):
        def isfile(path):
            try:
                st = os.stat(path)
            except OSError:
                return False
            return stat.S_ISREG(st.st_mode)
        all_weights_files = [f for f in os.listdir(path_to_weights_folder) if isfile(os.path.join(path_to_weights_folder, f))]
        if len(all_weights_files) == 0:
            raise ValueError(f"No weights in {path_to_weights_folder}")

        self.model.load_weights(os.path.join(path_to_weights_folder, all_weights_files[-1]))

    def save_model(self, path_to_dir):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(os.path.join(path_to_dir, "model.tflite", "wb")) as f:
            f.write(tflite_model)
        # tf.saved_model.save(self.model, path_to_dir)

        

    ######################
    ### Logs functions ###
    ######################

    ### Scalars

    @staticmethod
    def log(writer, get_metrics, reset_metrics, step):
        with writer.as_default():
            for name, value in get_metrics():
                tf.summary.scalar(name, value, step=step)
        reset_metrics()

    
    def log_train(self):
        Engine.log(self.train_writer, self.get_train_metrics, self.reset_train_metrics, self.i)


    def log_test(self):
        Engine.log(self.test_writer, self.get_test_metrics, self.reset_test_metrics, self.i)


    def log_images_test(self):
        grid = self.get_grid()
        columns = self.get_columns()

        with self.test_writer.as_default():
            tf.summary.image("Test output", Engine.build_images_grid(grid, columns), step=self.image_index)
        self.image_index += 1


    ### Images

    @staticmethod
    def plot_to_image(figure):
        # https://www.tensorflow.org/tensorboard/image_summaries
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)    
        return image

    @staticmethod
    def build_images_grid(grid, columns):
        n_images = len(grid)
        n_categories = len(grid[0])
        fig, axs = plt.subplots(n_images, n_categories)
        scale_img = 3
        fig.set_size_inches(n_categories*scale_img,n_images*scale_img)
        
        for line, row in zip(grid, range(n_images)):
            for im, col, name in zip(line, range(n_categories), columns):
                axs[row, col].imshow(im)
                axs[row, col].axis("off")
                if row == 0:
                    axs[0, col].set_title(name)
        return Engine.plot_to_image(fig)
    

    ############################################
    ### Methods to overload in child classes ###
    ############################################

    def compute_losses(self, y_true, y_pred):
        raise NotImplementedError()

    def get_train_metrics(self):
        raise NotImplementedError()

    def get_test_metrics(self):
        raise NotImplementedError()

    def reset_train_metrics(self):
        raise NotImplementedError()

    def reset_test_metrics(self):
        raise NotImplementedError()

    def update_train_metrics(self, losses):
        raise NotImplementedError()

    def update_test_metrics(self, losses):
        raise NotImplementedError()

    def extract_training_loss(self, losses):
        raise NotImplementedError()

    def get_grid(self):
        raise NotImplementedError()

    def get_columns(self):
        raise NotImplementedError()



    ##############################
    ### Conditions to overload ###
    ##############################

    def main_loop_condition(self):
        raise NotImplementedError()

    def training_loop_condition(self):
        raise NotImplementedError()

    def testing_condition(self):
        raise NotImplementedError()
    
    def reset_testing_condition(self):
        raise NotImplementedError()

    def saving_condition(self):
        raise NotImplementedError()

    def reset_saving_condition(self): 
        raise NotImplementedError()

    def training_log_condition(self):
        raise NotImplementedError()

    def reset_training_log_condition(self):
        raise NotImplementedError()

    def testing_log_condition(self):
        raise NotImplementedError()

    def reset_testing_log_condition(self):
        raise NotImplementedError()
