import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.gen_dataset_ops import TextLineDataset
import tensorflow_addons as tfa
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from os.path import join

class AdaMattingDataset:
    def __init__(self, mode, dataset_folder, batch_size=32, img_size = (1024, 1024)) -> None:
        self._size = img_size
        self._dataset_folder = dataset_folder
        self._mode = mode
        self._batch_size = batch_size

        self._root_folder = join(self._dataset_folder, self._mode)

        self._data_list_file = tf.data.TextLineDataset(join(self._root_folder, "data.csv"))
        self._df = self._data_list_file.map(lambda x : self.preprocess(x), num_parallel_calls=5).shuffle(500).batch(batch_size).prefetch(1)

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
                tf.cast(a <= 0, dtype=tf.float32),
                tf.cast((a > 0) & (a < 1), dtype=tf.float32),
                tf.cast(a >= 1, dtype=tf.float32)
            ], axis=-1)

        gt_trimap = extract_trimap(gt_alpha)

        # Merge inputs
        x = tf.concat([image, gen_trimap], axis=-1)

        return x, gt_trimap


    def show(self, input, n=3):
        x, y = input
        fig, axs = plt.subplots(n,3)
        for row in range(n):
            axs[row, 0].imshow(x[row, :,:,0:3])
            axs[row, 0].set_title(f"[{row}] Patched image")

            axs[row, 1].imshow(y[row, :,:,:])
            axs[row, 1].set_title(f"[{row}] Real Trimap")

            axs[row, 2].imshow(x[row,:,:,3:6])
            axs[row, 2].set_title(f"[{row}] User's trimap input")

        plt.legend()
        plt.show()

# Tests
# df = AdaMattingDataset("train", "/net/rnd/DEV/Datasets_DL/alpha_matting/")
# df.show(next(iter(df._df)))
