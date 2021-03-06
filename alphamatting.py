import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from PIL import Image
from network import get_model
from os.path import join
from os import listdir
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

##################
### Parametres ###
##################

path_weights = "/net/homes/r/rseailles/Deep/OwnAdaMatting/saves/FullAdaMatting/10-29_19h05/11-02_11h49.h5"
path_alphamatting = "/net/homes/r/rseailles/Deep/Alphamatting/"

reduct_factor = 4

############
### Code ###
############

depth=32
m = get_model(depth=depth)[0]
m.load_weights(path_weights)

for file in tqdm(listdir(join(path_alphamatting, "images/"))):
    def open_img (folder):
        img = tf.image.convert_image_dtype(tf.image.decode_image(tf.io.read_file(join(path_alphamatting, folder, file))), dtype="float32")
        x,y = int(img.shape[0]/(depth*reduct_factor))*depth, int(img.shape[1]/(depth*reduct_factor))*depth
        x,y = 512, 512
        img = tf.image.resize(img, (x, y))
        return img

    img = open_img("images/")
    tri = tf.slice(open_img("trimaps/"), [0,0,0], [-1, -1, 1])

    def extract_trimap(a):
        return tf.concat([
            tf.cast(a <= 0.01, dtype=tf.float32),
            tf.cast((a > 0.01) & (a < 0.95), dtype=tf.float32),
            tf.cast(a >= 0.95, dtype=tf.float32)
        ], axis=-1)

    tri = extract_trimap(tri)

    x = tf.concat([img, tri], axis=-1)
    y = tf.squeeze(m(tf.expand_dims(x, axis=0))[1], axis=0)
    y = tf.clip_by_value(y, 0.0, 1.0)

    background = tf.concat([
        tf.ones(shape=y.shape),
        tf.zeros(shape=y.shape),
        tf.zeros(shape=y.shape)
    ], axis=-1)

    alpha = tf.repeat(y, 3, axis=-1)
    composed = x[:,:,0:3]*alpha + background*(1.0-alpha)

    plt.imsave(join(path_alphamatting, "alphas/", file), tf.squeeze(y, axis=-1).numpy())
    plt.imsave(join(path_alphamatting, "composed/", file), composed.numpy())
    





