import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from os.path import join

class AdaMattingDataset:
    def __init__(self, mode, dataset_folder, img_size = (1024, 1024)) -> None:
        self._size = img_size
        self._size_tf = (img_size[1], img_size[0]) # Tensorflow convension

        self._aspect_ratio = self._size[0]/self._size[1]
        self._dataset_folder = dataset_folder
        self._dataset_folder_tf = tf.constant(dataset_folder)
        self._mode = mode

        self._path_data = join(dataset_folder, mode, "data.csv")
        self._dataset = tf.data.TextLineDataset(self._path_data).skip(1).shuffle(15000)

        def parse_csv(line):
            cols_types = [[""]] * 3  # all required
            columns = tf.io.decode_csv(line, record_defaults=cols_types, field_delim=',')
            return tf.stack(columns)

        self._dataset = self._dataset.map(parse_csv)

    def adapt(self, img):
        deltaH = self._size_tf[0] - img.shape[-3]
        deltaW = self._size_tf[1] - img.shape[-2]
        if deltaH > 0:
            img = tf.image.pad_to_bounding_box(img, int(deltaH/2), 0, self._size_tf[0], img.shape[-2])
        if deltaW > 0:
            img = tf.image.pad_to_bounding_box(img, 0, int(deltaW/2), img.shape[-3], self._size_tf[1])
        if deltaH < 0 or deltaW < 0:
            img = tf.image.resize_with_pad(img, self._size_tf[0], self._size_tf[1], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return img

    def gen_trimap(self, alpha):
        have_batch = (len(alpha.shape) == 4)
        if have_batch:
            alpha_exp = alpha
        else:
            alpha_exp = tf.expand_dims(alpha, axis=0)

        kernel_size_d = tf.random.uniform(shape=[2], minval=0, maxval=50, dtype=tf.int64)
        kernel_size_e = tf.random.uniform(shape=[2], minval=0, maxval=20, dtype=tf.int64)

        dilated = tf.nn.max_pool2d(alpha_exp, ksize=kernel_size_d, strides=1, padding="SAME")
        eroded = -tf.nn.max_pool2d(-alpha_exp, ksize=kernel_size_e, strides=1, padding="SAME")
        
        trimap = tf.concat([
            tf.cast(dilated <= 0, dtype=tf.float32),
            tf.cast((dilated > 0) & (eroded < 1), dtype=tf.float32),
            tf.cast(eroded >= 1, dtype=tf.float32)
        ], axis=-1)

        if have_batch:
            return trimap
        else:
            return tf.squeeze(trimap, axis=0)

    def extract_trimap(self, alpha):
        return tf.concat([
            tf.cast(alpha <= 0, dtype=tf.float32),
            tf.cast((alpha > 0) & (alpha < 1), dtype=tf.float32),
            tf.cast(alpha >= 1, dtype=tf.float32)
        ], axis=-1)

    def get_transf(self, init_fg_size):
        def get_rand(size_bg, size_fg):
            if size_fg == size_bg:
                return tf.constant(0)
            return tf.cast(tf.random.uniform(shape=[], minval=-int(abs(size_bg - size_fg)/2), maxval= int(abs(size_bg - size_fg)/2), dtype=tf.int64), dtype=tf.float32)
        return  tf.concat([get_rand(self._size_tf[1], init_fg_size[-2]), get_rand(self._size_tf[0], init_fg_size[-3])], axis=0)
        

    def process(self, line):
        # Ouverture des 3 images
        def open_img (path) : 
            return tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(tf.strings.join([self._dataset_folder_tf, path]))), dtype=float)
        
        fg_img =        open_img(line[0])
        alpha_img =     open_img(line[1])
        bg_img =        open_img(line[2])

        # Redimensionnement
        initial_fg_size = fg_img.shape
        bg_img      =   tf.image.resize(bg_img, self._size_tf)
        fg_img      =   self.adapt(fg_img)
        alpha_img   =   self.adapt(alpha_img)

        # Deplacement aleatoire
        deplacement =   self.get_transf(initial_fg_size)
        fg_img      =   tfa.image.translate(fg_img, translations=deplacement, fill_mode="constant")
        alpha_img   =   tfa.image.translate(alpha_img, translations=deplacement, fill_mode="constant")
        
        # Generating trimap
        x = [   (fg_img*alpha_img + bg_img*(1.0 - alpha_img)),
                self.gen_trimap(alpha_img)]

        # Retour
        return x, self.extract_trimap(alpha_img)

    def show(self, line, with_prediction=False):
        x, y = self.process(line)
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(x[0])
        axs[0].set_title("Patched image")

        axs[1].imshow(y)
        axs[1].set_title("Real Trimap")

        axs[2].imshow(x[1])
        axs[2].set_title("User's trimap input")

        plt.legend()
        plt.show()

    def get_dataset(self, batch_size, threads=5, shuffle_buffer=10000, repeat=5):
        self._dataset = self._dataset.shuffle(shuffle_buffer).repeat(repeat)
        self._dataset = self._dataset.map(lambda line : self.process(line), num_parallel_calls=threads)
        return self._dataset.batch(batch_size=batch_size).prefetch(1)


# Tests
# df = AdaMattingDataset("train", "/net/rnd/DEV/Datasets_DL/alpha_matting/")
# df.show(next(iter(df._dataset)))