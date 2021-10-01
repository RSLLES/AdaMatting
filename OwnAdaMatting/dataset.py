import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.ops import gen_nn_ops

from os.path import join
from os import listdir

# class AdaMattingDataset:
#     def __init__(self, mode, dataset_folder, batch_size=32, img_size = (1024, 1024), shuffle_buffer = 15000) -> None:
#         self._size = img_size
#         self._dataset_folder = dataset_folder
#         self._mode = mode
#         self._root_folder = join(self._dataset_folder, self._mode)

#         self._n_test = 100
#         self._n_val = 5

#         self._batch_size = batch_size
#         self._autotune = tf.data.experimental.AUTOTUNE

#         #  Initial load and shuffle
#         self._data_list_file = tf.data.TextLineDataset(join(self._root_folder, "data.csv")).shuffle(shuffle_buffer, reshuffle_each_iteration=False)

#         #  Split
#         self._df_test = self._data_list_file.take(self._n_test + self._n_val)
#         self._df_train = self._data_list_file.skip(self._n_test + self._n_val)

#         self._df_val = self._df_test.take(self._n_val)
#         self._df_test = self._df_test.skip(self._n_val)

#         # Preprocess
#         self._df_train = self._df_train.map(lambda x : self.preprocess(x), num_parallel_calls=self._autotune).batch(batch_size).prefetch(self._autotune)
#         self._df_test = self._df_test.map(lambda x : self.preprocess(x), num_parallel_calls=self._autotune).batch(batch_size).prefetch(self._autotune)
#         self._df_val = self._df_val.map(lambda x : self.preprocess(x), num_parallel_calls=self._autotune).batch(1).prefetch(self._autotune)


#     def preprocess(self, line):
#         def open_img(root_path, file):
#             return  tf.image.resize(
#                         tf.image.convert_image_dtype(
#                             tf.image.decode_jpeg(
#                                 tf.io.read_file(
#                                     tf.strings.join([root_path, file])
#                                 )
#                             ), dtype=float
#                         ), (self._size[1], self._size[0])
#                     )

#         image = open_img(join(self._root_folder, "images/"), line)
#         gt_alpha = open_img(join(self._root_folder, "gt_alpha/"), line)
#         gen_trimap = open_img(join(self._root_folder, "gen_trimap/"), line)

#         # Extraction de la trimap depuis l'alpha
#         def extract_trimap(a):
#             return tf.concat([
#                 tf.cast(a <= 0.01, dtype=tf.float32),
#                 tf.cast((a > 0.01) & (a < 0.95), dtype=tf.float32),
#                 tf.cast(a >= 0.95, dtype=tf.float32)
#             ], axis=-1)

#         gt_trimap = extract_trimap(gt_alpha)

#         # Merge inputs and outputs
#         x = tf.concat([image, gen_trimap], axis=-1)
#         y = tf.concat([gt_trimap, gt_alpha], axis=-1)

#         return x, y


class LiveComputedDataset:
    def __init__(self, mode, dataset_folder, batch_size=32, img_size = (512, 512), shuffle_buffer = 15000) -> None:
        self._size = img_size
        self._size_tf = tf.constant([self._size[1], self._size[0], 3+3+3])
        self._dataset_folder = dataset_folder
        self._mode = mode
        self._root_folder = join(self._dataset_folder, self._mode)

        self._alpha_folder = tf.constant(join(self._root_folder, "alpha/"))
        self._fg_folder = tf.constant(join(self._root_folder, "fg/"))
        self._bg_folder = tf.constant(join(self._root_folder, "bg/"))

        self._n_test = 10
        self._n_images = 5

        self._batch_size = batch_size
        self._autotune = tf.data.experimental.AUTOTUNE

        self._ds_fg_files = tf.data.Dataset.from_tensor_slices(listdir(join(self._root_folder, "fg/")))
        self._ds_bg_files = tf.data.Dataset.from_tensor_slices(listdir(join(self._root_folder, "bg/")))
        self._ds_val_files = tf.data.Dataset.from_tensor_slices(listdir("/net/homes/r/rseailles/Deep/Alphamatting/alphas/"))

        self._ds_test_fg_files = self._ds_fg_files.take(self._n_test)
        self._ds_test_bg_files = self._ds_bg_files.take(self._n_test)
        self._ds_train_fg_files = self._ds_fg_files.skip(self._n_test).shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        self._ds_train_bg_files = self._ds_bg_files.skip(self._n_test).shuffle(shuffle_buffer, reshuffle_each_iteration=True)

        self._ds_train = tf.data.Dataset.zip((self._ds_train_fg_files, self._ds_train_bg_files))
        self._ds_test = tf.data.Dataset.zip((self._ds_test_fg_files, self._ds_test_bg_files))

        self._ds_train = self._ds_train.map(lambda x1, x2 : self.preprocess(x1,x2), num_parallel_calls = self._autotune).batch(batch_size).prefetch(self._autotune)
        self._ds_test = self._ds_test.map(lambda x1, x2 : self.preprocess(x1,x2), num_parallel_calls = self._autotune).batch(1)
        self._ds_val = self._ds_val_files.map(lambda x : self.preprocess_val(x), num_parallel_calls = self._autotune).batch(1)

        
    def preprocess_val(self, file):
        path_alphamatting = tf.constant("/net/homes/r/rseailles/Deep/Alphamatting/")
        path_image = tf.strings.join([path_alphamatting, tf.constant("images/"), file])
        path_trimap = tf.strings.join([path_alphamatting, tf.constant("trimaps/"), file])

        import_img = lambda path : tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(path)), dtype="float32")

        img = import_img(path_image)
        tri = tf.slice(import_img(path_trimap), [0,0,0], [-1, -1, 1])

        rounded_size = tf.math.multiply(tf.cast(tf.math.divide(tf.shape(img)[-3:-1], 16), dtype="int32"), 16)
        img = tf.image.resize(img, rounded_size)
        tri = tf.image.resize(tri, rounded_size)

        def change_trimap_format(a):
            return tf.concat([
                tf.cast(a <= 0.01, dtype=tf.float32),
                tf.cast((a > 0.01) & (a < 0.95), dtype=tf.float32),
                tf.cast(a >= 0.95, dtype=tf.float32)
            ], axis=-1)
        tri = change_trimap_format(tri)

        return tf.concat([img, tri], axis=-1)



    def preprocess(self, fg_file, bg_file):
        # Importation
        import_img = lambda path : tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(path)), dtype="float32")
        gt_alpha = import_img(tf.strings.join([self._alpha_folder, fg_file]))
        gt_alpha = tf.slice(gt_alpha, [0,0,0], [-1, -1, 1]) # Safety
        fg = import_img(tf.strings.join([self._fg_folder, fg_file]))
        bg = import_img(tf.strings.join([self._bg_folder, bg_file]))

        # Angle, Flip et rescale
        angle = tf.random.uniform(shape=[], minval=0.0, maxval=2*3.1416)
        gt_alpha = tfa.image.rotate(gt_alpha, angle)
        fg = tfa.image.rotate(fg, angle)

        # gt_alpha = tf.image.resize(gt_alpha, self._size_tf[0:2])
        # fg = tf.image.resize(fg, self._size_tf[0:2])
        # bg = tf.image.resize(bg, self._size_tf[0:2])

        # Adaptation taille to patch
        # fg = tf.image.resize_with_crop_or_pad(fg, tf.shape(bg)[0], tf.shape(bg)[1])
        # gt_alpha = tf.image.resize_with_crop_or_pad(gt_alpha, tf.shape(bg)[0], tf.shape(bg)[1])
        bg = tf.image.resize(bg, tf.shape(fg)[:2])

        # Position
        limit = tf.math.divide(tf.shape(bg)[:2], 3)
        depl =  tf.random.uniform(
            shape=limit.shape, 
            minval=-limit, 
            maxval=limit, 
            dtype="float64"
            )
        depl = tf.cast(depl, dtype="int32")

        fg = tfa.image.translate_xy(fg, depl, replace=0)
        gt_alpha = tf.repeat(gt_alpha, repeats=3, axis=-1)
        gt_alpha = tfa.image.translate_xy(gt_alpha, depl, replace=0)

        # Random Crop
        all = tf.concat([fg, bg, gt_alpha], axis=-1)
        if tf.reduce_all(self._size_tf <= tf.shape(all) ):
            all = tf.image.random_crop(all, size=self._size_tf)
        else:
            all = tf.image.resize(all, self._size_tf[0:2])
        fg = tf.slice(all, [0,0,0],[-1,-1,3])
        bg = tf.slice(all, [0,0,3],[-1,-1,3])
        gt_alpha = tf.slice(all, [0,0,6],[-1,-1,3])

        # Patch
        img = fg*gt_alpha + bg*(1.0-gt_alpha)

        # Brightness and Contrast
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.clip_by_value(img, 0.0, 1.0)

        # Gen Trimap
        @tf.function
        def random_dilate(dilated, eroded):
            kernel_sizes =  tf.random.uniform(
                shape=[4], 
                minval=3, 
                maxval=30, 
                dtype="int32"
                )
            dilated = gen_nn_ops.max_pool_v2(dilated, [1, kernel_sizes[0], kernel_sizes[1], 1], [1, 1, 1, 1], "SAME")
            eroded = -gen_nn_ops.max_pool_v2(-eroded, [1, kernel_sizes[2], kernel_sizes[3], 1], [1, 1, 1, 1], "SAME")
            return dilated, eroded
        
        @tf.function
        def constant_dilate(dilated, eroded, strengh=3):
            # dilated = gen_nn_ops.max_pool_v2(dilated, [1, strengh, strengh, 1], [1, 1, 1, 1], "SAME")
            # eroded = -gen_nn_ops.max_pool_v2(-eroded, [1, strengh, strengh, 1], [1, 1, 1, 1], "SAME")
            return dilated, eroded

        @tf.function
        def build_trimap(gt_alpha, func, i=tf.constant(1, dtype="int32")):
            dilated, eroded = gt_alpha, gt_alpha
            if tf.less(0, i):
                dilated, eroded = func(dilated, eroded)
            if tf.less(1, i):
                dilated, eroded = func(dilated, eroded)
            if tf.less(2, i):
                dilated, eroded = func(dilated, eroded)
            if tf.less(3, i):
                dilated, eroded = func(dilated, eroded)
            if tf.less(4, i):
                dilated, eroded = func(dilated, eroded)

            trimap_bg = tf.cast(dilated <= 0.1, dtype="float32")
            trimap_fg = tf.cast(eroded >= 0.95, dtype="float32")
            trimap_uk = 1.0 - trimap_bg - trimap_fg + trimap_bg*trimap_fg
            return tf.concat([trimap_bg, trimap_uk, trimap_fg], axis=-1)
    

        gt_alpha = tf.expand_dims(tf.slice(gt_alpha, [0,0,0], [-1,-1,1]), axis=0)

        gen_trimap = build_trimap(gt_alpha, random_dilate, i = tf.random.uniform(shape=[], minval=1, maxval=5, dtype="int32"))
        gt_trimap = build_trimap(gt_alpha, constant_dilate)

        gen_trimap = tf.squeeze(gen_trimap, axis=0)
        gt_trimap = tf.squeeze(gt_trimap, axis=0)
        gt_alpha = tf.squeeze(gt_alpha, axis=0)

        return tf.concat([img, gen_trimap], axis=-1), tf.concat([gt_trimap, gt_alpha], axis=-1)

# df = LiveComputedDataset("picky", "/net/rnd/DEV/Datasets_DL/alpha_matting/", img_size=(512, 512), batch_size=1)
# fg = next(iter(df._ds_test_fg_files))
# bg = next(iter(df._ds_test_bg_files))
# a = df.preprocess(fg, bg)
# print(a)