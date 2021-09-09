import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

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
        iterations = np.random.randint(1, 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(alpha, kernel, iterations)
        eroded = cv2.erode(alpha, kernel, iterations)
        trimap = np.zeros(alpha.shape)
        trimap.fill(128)
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0
        return trimap

    def process(self, line):
        # Ouverture des 3 images
        fg_img = cv2.imread(join(self._dataset_folder, line[0].numpy().decode('UTF-8')), -1)
        alpha_img = cv2.imread(join(self._dataset_folder, line[1].numpy().decode('UTF-8')), -1)
        bg_img = cv2.imread(join(self._dataset_folder, line[2].numpy().decode('UTF-8')), -1)

        # Redimensionnement
        bg_img = cv2.resize(bg_img, dsize=self._size, interpolation=cv2.INTER_NEAREST)
        fg_img      =   self.adapt(fg_img)
        alpha_img   =   self.adapt(alpha_img)

        # Deplacement aleatoire
        M = np.array((
            (1, 0,  np.random.randint(low = -self._size[0]/2, high=self._size[0]/2)),
            (0, 1,  np.random.randint(low = -self._size[1]/2, high=self._size[1]/2))
        ), dtype=np.float32)
        bg_img      =   bg_img
        fg_img      =   cv2.warpAffine(fg_img, M, self._size)
        alpha_img   =   cv2.warpAffine(alpha_img, M, self._size)
        alpha_img_norm   =   np.expand_dims(alpha_img, axis=-1).astype(np.float)/255
        
        # Generating trimap

        x = fg_img*alpha_img_norm + bg_img*(1.0 - alpha_img_norm)

        # Retour
        return x.astype(np.uint8)


# Tests
df = AdaMattingDataset("train", "/net/rnd/DEV/Datasets_DL/alpha_matting/")
cv2.imshow("test", df.process(next(iter(df._dataset))))
cv2.waitKey(0)