import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from os.path import join

class AdaMattingDataset:
    def __init__(self, mode, dataset_folder) -> None:
        self._size = (1280, 720)
        self._aspect_ratio = self._size[0]/self._size[1]
        self._dataset_folder = dataset_folder
        self._mode = mode

        self._path_data = join(dataset_folder, mode, "data.csv")
        self._dataset = tf.data.TextLineDataset(self._path_data).skip(1).shuffle(15000)

        def parse_csv(line):
            cols_types = [[""]] * 3  # all required
            columns = tf.io.decode_csv(line, record_defaults=cols_types, field_delim=',')
            return tf.stack(columns)

        self._dataset = self._dataset.map(parse_csv)

    def adapt(self, img):
        deltaW = self._size[0] - img.shape[1] # Shape is (rows, cols) which in image language is (height, weidht)
        deltaH = self._size[1] - img.shape[0]
        if deltaW > 0:
             img = cv2.copyMakeBorder(img, 0, 0, int(deltaW/2), deltaW - int(deltaW/2), cv2.BORDER_CONSTANT)
        if deltaH > 0:
             img = cv2.copyMakeBorder(img, int(deltaH/2), deltaH - int(deltaH/2), 0, 0, cv2.BORDER_CONSTANT)
        if deltaH < 0 or deltaW < 0:
            cv2.resize(img, self._size, interpolation=cv2.INTER_NEAREST)
        return img

    def gen_trimap(self, alpha):
        k_size = np.random.choice(range(1, 5))
        iterations_d, iterations_e = np.random.randint(1, 15), np.random.randint(1, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(alpha, kernel, iterations_d)
        eroded = cv2.erode(alpha, kernel, iterations_e)
        
        trimap = np.zeros((alpha.shape[0], alpha.shape[1], 3), dtype=np.float32)
        trimap[:, :, 0][dilated <= 0] = 1.0
        trimap[:, :, 1][(dilated > 0) & (eroded < 1.0)] = 1.0
        trimap[:, :, 2][eroded >= 1.0] = 1.0
        return trimap

    def get_transf(self, init_fg_size):
        def get_rand(size_bg, size_fg):
            if size_fg == size_bg:
                return 0
            return np.random.randint(   low = -int(abs(size_bg - size_fg)/2), 
                                        high=  int(abs(size_bg - size_fg)/2))
        return  np.array((
                    (1, 0,  get_rand(self._size[0], init_fg_size[0])),
                    (0, 1,  get_rand(self._size[1], init_fg_size[1]))
                ), dtype=np.float32)
        

    def process(self, line):
        # Ouverture des 3 images
        fg_img = cv2.imread(join(self._dataset_folder, line[0].numpy().decode('UTF-8')), -1)
        alpha_img = cv2.imread(join(self._dataset_folder, line[1].numpy().decode('UTF-8')), -1)
        bg_img = cv2.imread(join(self._dataset_folder, line[2].numpy().decode('UTF-8')), -1)

        # Redimensionnement
        initial_fg_size = fg_img.shape
        bg_img = cv2.resize(bg_img, dsize=self._size, interpolation=cv2.INTER_NEAREST)
        fg_img      =   self.adapt(fg_img)
        alpha_img   =   self.adapt(alpha_img)

        # Deplacement aleatoire
        M = self.get_transf((initial_fg_size[1], initial_fg_size[0]))
        bg_img      =   bg_img.astype(np.float32)/255
        fg_img      =   cv2.warpAffine(fg_img, M, self._size).astype(np.float32)/255
        alpha_img   =   cv2.warpAffine(alpha_img, M, self._size).astype(np.float32)/255
        alpha_img_norm   =   np.expand_dims(alpha_img, axis=-1)
        
        # Generating trimap
        x = [   (fg_img*alpha_img_norm + bg_img*(1.0 - alpha_img_norm)),
                self.gen_trimap(alpha_img)]

        # Retour
        return x, alpha_img

    def show(self, line, with_prediction=False):
        x, y = self.process(line)
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(x[0][:,:,[2,1,0]]) # OpenCV uses BGR convention but Matplotlib uses RGB
        axs[0].set_title("Patched image")

        axs[1].imshow(y)
        axs[1].set_title("Real Alpha")

        axs[2].imshow(x[1])
        axs[2].set_title("User's trimap input")

        plt.legend()
        plt.show()


# Tests
df = AdaMattingDataset("train", "/net/rnd/DEV/Datasets_DL/alpha_matting/")
df.show(next(iter(df._dataset)))