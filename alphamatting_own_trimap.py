import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from PIL import Image
from network import get_model
from os.path import join
from os import listdir
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from dataset import DeepDataset

##################
### Parametres ###
##################

weights = "10-15_12h15/10-18_09h08.h5"

path_alphamatting = "/net/homes/r/rseailles/Deep/AlphaMatting_GenTrimap"
path_weights = "/net/homes/r/rseailles/Deep/OwnAdaMatting/saves/"

############
### Code ###
############
df = DeepDataset("/net/rnd/DEV/Datasets_DL/alpha_matting/alphamatting_test", img_size=(640,640))

depth=32
m, _ = get_model(None, depth=depth)
m.load_weights(join(path_weights, weights))

i = 0
for x, _ in tqdm(df._ds_test):
    trimap_adapted , alpha, _ = m(x)
    alpha = tf.squeeze(alpha, axis=0)
    alpha = tf.clip_by_value(alpha, 0.0, 1.0)
    trimap_adapted = tf.squeeze(trimap_adapted, axis=0)
    trimap_adapted = tf.clip_by_value(trimap_adapted, 0.0, 1.0)

    img = tf.squeeze(tf.slice(x, [0,0,0,0], [1,-1,-1,3]), axis=0)
    tri = tf.squeeze(tf.slice(x, [0,0,0,3], [1,-1,-1,3]), axis=0)

    background = tf.concat([
        tf.ones(shape=alpha.shape),
        tf.zeros(shape=alpha.shape),
        tf.zeros(shape=alpha.shape)
    ], axis=-1)

    alpha = tf.repeat(alpha, 3, axis=-1)
    composed = img*alpha + background*(1.0-alpha)

    plt.imsave(join(path_alphamatting, "images/", f"{i}.png"), img.numpy())
    plt.imsave(join(path_alphamatting, "alphas/", f"{i}.png"), alpha.numpy())
    plt.imsave(join(path_alphamatting, "composed/", f"{i}.png"), composed.numpy())
    plt.imsave(join(path_alphamatting, "trimap/", f"{i}.png"), tri.numpy())
    plt.imsave(join(path_alphamatting, "trimap_refined/", f"{i}.png"), trimap_adapted.numpy())
    i+=1
    





