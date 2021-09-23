import numpy as np
import pandas as pd

import tensorflow as tf

# from keras.utils.vis_utils import plot_model

from os.path import join

class AdaMattingDataset:
    def __init__(self, mode, dataset_folder, batch_size=32, img_size = (1024, 1024), shuffle_buffer = 15000) -> None:
        self._size = img_size
        self._dataset_folder = dataset_folder
        self._mode = mode
        self._root_folder = join(self._dataset_folder, self._mode)

        self._n_test = 100
        self._n_val = 5

        self._batch_size = batch_size
        self._autotune = tf.data.experimental.AUTOTUNE

        #  Initial load and shuffle
        self._data_list_file = tf.data.TextLineDataset(join(self._root_folder, "data.csv")).shuffle(shuffle_buffer, reshuffle_each_iteration=False)

        #  Split
        self._df_test = self._data_list_file.take(self._n_test + self._n_val)
        self._df_train = self._data_list_file.skip(self._n_test + self._n_val)

        self._df_val = self._df_test.take(self._n_val)
        self._df_test = self._df_test.skip(self._n_val)

        # Preprocess
        self._df_train = self._df_train.map(lambda x : self.preprocess(x), num_parallel_calls=self._autotune).batch(batch_size).prefetch(self._autotune)
        self._df_test = self._df_test.map(lambda x : self.preprocess(x), num_parallel_calls=self._autotune).batch(batch_size).prefetch(self._autotune)
        self._df_val = self._df_val.map(lambda x : self.preprocess(x), num_parallel_calls=self._autotune).batch(1).prefetch(self._autotune)


    def preprocess(self, line):
        def open_img(root_path, file):
            return  tf.image.resize(
                        tf.image.convert_image_dtype(
                            tf.image.decode_jpeg(
                                tf.io.read_file(
                                    tf.strings.join([root_path, file])
                                )
                            ), dtype=float
                        ), (self._size[1], self._size[0])
                    )

        image = open_img(join(self._root_folder, "images/"), line)
        gt_alpha = open_img(join(self._root_folder, "gt_alpha/"), line)
        gen_trimap = open_img(join(self._root_folder, "gen_trimap/"), line)

        # Extraction de la trimap depuis l'alpha
        def extract_trimap(a):
            return tf.concat([
                tf.cast(a <= 0.01, dtype=tf.float32),
                tf.cast((a > 0.01) & (a < 0.95), dtype=tf.float32),
                tf.cast(a >= 0.95, dtype=tf.float32)
            ], axis=-1)

        gt_trimap = extract_trimap(gt_alpha)

        # Merge inputs and outputs
        x = tf.concat([image, gen_trimap], axis=-1)
        y = tf.concat([gt_trimap, gt_alpha], axis=-1)

        return x, y

