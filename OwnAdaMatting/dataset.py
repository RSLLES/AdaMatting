from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.ops import gen_nn_ops

from os.path import join
from os import listdir
from os.path import isfile  

class DeepDataset:
    def __init__(self, dataset_folder, batch_size=32, img_size = (320, 320), size_dividor=32, max_size_factor=2, shuffle_buffer = 150000) -> None:
        assert img_size[0]%size_dividor == 0 and img_size[1]%size_dividor == 0
        assert img_size[0] == img_size[1]
        self._size = img_size[0]
        self._size_tf = tf.constant([self._size, self._size])
        self._size_dividor = size_dividor
        self._min_size_factor = int(img_size[0]/size_dividor)
        self._max_size_factor = self._min_size_factor*max_size_factor
        self._sizes_crop = tf.constant([[self._size_dividor*k, self._size_dividor*k] for k in range(self._min_size_factor, self._max_size_factor+1)], dtype="int32")
        self._sizes_crop_proba = tf.expand_dims(tf.constant([1/k for k in range(self._min_size_factor, self._max_size_factor+1)], dtype="float32"), axis=0)
        self._batch_size = batch_size
        self._n_images = 5

        assert isfile(join(dataset_folder, "train.csv")) or isfile(join(dataset_folder, "test.csv"))
        self._dataset_folder = dataset_folder
        self._train_csv = join(dataset_folder, "train.csv")
        self._test_csv = join(dataset_folder, "test.csv")

        self._laplacian_kernel = tf.expand_dims(tf.expand_dims(
            tf.constant([
                [1,  1, 1],
                [1, -8, 1],
                [1,  1, 1]
            ], dtype="float32")*tf.constant((0.01), dtype="float32"), axis=-1), axis=-1)
        self._heat_equation_size_alone = int(self._size/4)
        self._heat_equation_size = tf.constant([self._heat_equation_size_alone, self._heat_equation_size_alone], dtype="int32")
        self._nb_iterations = 50

        self._padding_cst = 8
        self._central_crop = 1 - self._padding_cst*8/self._size
        self._padding_window = tf.constant([self._padding_cst, self._padding_cst])

        self._autotune = tf.data.experimental.AUTOTUNE
        if isfile(join(dataset_folder, "train.csv")):
            self._ds_train_files = tf.data.experimental.make_csv_dataset(
                self._train_csv,
                column_names=["fg", "alpha", "bg"],
                batch_size=1,
                header=False,
                num_epochs=1).shuffle(shuffle_buffer, reshuffle_each_iteration=True)
            self._ds_train = self._ds_train_files.map(lambda x : self.preprocess(x), num_parallel_calls = self._autotune).batch(batch_size).prefetch(self._autotune)

        if isfile(join(dataset_folder, "test.csv")):
            with open(join(dataset_folder, "test.csv")) as f:
                assert batch_size*self._n_images <= len(f.readlines())
            self._ds_test_files = tf.data.experimental.make_csv_dataset(
                self._test_csv, 
                column_names=["fg", "alpha", "bg"],
                batch_size=1,
                header=False,
                num_epochs=1)
            self._ds_test = self._ds_test_files.map(lambda x : self.preprocess(x), num_parallel_calls = self._autotune).batch(batch_size).prefetch(self._autotune)

    
    def preprocess(self, line):
        # Importation
        import_img = lambda type_img : tf.image.convert_image_dtype(tf.image.decode_png(tf.io.read_file(tf.squeeze(line[type_img]))), dtype="float32")

        gt_alpha = import_img("alpha")
        fg = import_img("fg")
        bg = import_img("bg")
        gt_alpha = tf.slice(gt_alpha, [0,0,0], [-1, -1, 1]) # Safety

        # Angle and Flip
        # angle = tf.random.uniform(shape=[], minval=-0.1*3.1416, maxval=0.1*3.1416)
        # gt_alpha = tfa.image.rotate(gt_alpha, angle)
        # fg = tfa.image.rotate(fg, angle)

        fg_and_alpha = tf.concat([fg, gt_alpha], axis=-1)
        fg_and_alpha = tf.image.random_flip_left_right(fg_and_alpha)
        # fg_and_alpha = tf.image.random_flip_up_down(fg_and_alpha)
        fg = tf.slice(fg_and_alpha, [0,0,0], [-1,-1,3])
        gt_alpha = tf.slice(fg_and_alpha, [0,0,3], [-1,-1,1])

        # Pick a grey pixel and center around it
        trimap_bg = tf.cast(gt_alpha <= 0.01, dtype="float32")
        trimap_fg = tf.cast(gt_alpha >= 0.99, dtype="float32")
        unknown_zone = 1.0 - trimap_bg - trimap_fg + trimap_bg*trimap_fg
        unknown_zone = tf.reshape(unknown_zone, [1, tf.shape(unknown_zone)[0]*tf.shape(unknown_zone)[1]])
        rand_pixel_flatten = tf.squeeze(tf.cast(tf.random.categorical(tf.math.log(unknown_zone), 1), dtype="int32"))
        rand_pixel = tf.stack([
            tf.cast(tf.math.divide(rand_pixel_flatten, tf.shape(gt_alpha)[1]), dtype="int32"),
            tf.math.floormod(rand_pixel_flatten, tf.shape(gt_alpha)[1])
            ])
        rand_pixel = tf.clip_by_value(
            rand_pixel, 
            clip_value_min = self._padding_window, 
            clip_value_max = tf.shape(gt_alpha)[:2] - self._padding_window)

        # Crop and resize
        croping_size = self._sizes_crop[
            tf.squeeze(tf.random.categorical(tf.math.log(self._sizes_crop_proba), 1))
        ]
        
        # Adjusting foreground
        half = tf.cast(tf.math.divide(croping_size, 2), dtype="int32")
        d_hg = -tf.clip_by_value(rand_pixel - half, -10000, 0)
        d_bd = -tf.clip_by_value(tf.shape(fg)[:2] - rand_pixel - half, -10000, 0)

        fg = tf.image.pad_to_bounding_box(
            fg, 
            d_hg[0], 
            d_hg[1], 
            tf.shape(fg)[0] + d_bd[0] + d_hg[0], 
            tf.shape(fg)[1] + d_bd[1] + d_hg[1])

        fg = tf.image.crop_to_bounding_box(
            fg, 
            rand_pixel[0] + d_hg[0] - half[0], 
            rand_pixel[1] + d_hg[1] - half[1], 
            croping_size[0], 
            croping_size[1])

        gt_alpha = tf.image.pad_to_bounding_box(
            gt_alpha, 
            d_hg[0], 
            d_hg[1], 
            tf.shape(gt_alpha)[0] + d_bd[0] + d_hg[0], 
            tf.shape(gt_alpha)[1] + d_bd[1] + d_hg[1])

        gt_alpha = tf.image.crop_to_bounding_box(
            gt_alpha, 
            rand_pixel[0] + d_hg[0] - half[0], 
            rand_pixel[1] + d_hg[1] - half[1], 
            croping_size[0], 
            croping_size[1])
        
        bg = tf.image.resize(bg, croping_size)

        # Patch
        img = fg*gt_alpha + bg*(1.0-gt_alpha)

        # Brightness, Contrast, Hue
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_saturation(img, 0.7, 1.5)
        # img = tf.image.random_hue(img, 0.)
        img = tf.clip_by_value(img, 0.0, 1.0)

        # Reduce
        img = tf.image.resize(img, self._size_tf[:2])
        gt_alpha = tf.image.resize(gt_alpha, self._size_tf[:2])

        # Expand Alpha
        gt_alpha = tf.expand_dims(tf.slice(gt_alpha, [0,0,0], [-1,-1,1]), axis=0)

        # Ground Truth Trimap
        @tf.function
        def build_gt_trimap(gt_alpha):
            trimap_bg = tf.cast(gt_alpha <= 0.01, dtype="float32")
            trimap_fg = tf.cast(gt_alpha >= 0.99, dtype="float32")
            trimap_uk = 1.0 - trimap_bg - trimap_fg + trimap_bg*trimap_fg
            return tf.concat([trimap_bg, trimap_uk, trimap_fg], axis=-1)

        gt_trimap = build_gt_trimap(gt_alpha)

        # Gen Trimap
        @tf.function
        def build_initial_condition(gt_alpha, strengh=10): # Strengh should be > 8 (because right after this, the resize operation divide the image dimensions by 8
            dilated = gen_nn_ops.max_pool_v2(gt_alpha, [1, strengh, strengh, 1], [1, 1, 1, 1], "SAME")
            eroded = -gen_nn_ops.max_pool_v2(-gt_alpha, [1, strengh, strengh, 1], [1, 1, 1, 1], "SAME")

            trimap_bg = tf.cast(dilated <= 0.1, dtype="float32")
            trimap_fg = tf.cast(eroded >= 0.95, dtype="float32")
            trimap_uk = 1.0 - trimap_bg - trimap_fg + trimap_bg*trimap_fg
            return trimap_uk
        
        u_ori = build_initial_condition(gt_alpha, strengh=10)

        # Resize and pad before use
        u_ori = tf.pad(
            tf.image.resize(u_ori, self._heat_equation_size), 
            paddings=tf.constant(
                [
                    [0,0],
                    [self._padding_cst,self._padding_cst],
                    [self._padding_cst,self._padding_cst],
                    [0,0]
                ]), 
            mode="SYMMETRIC")

        @tf.function
        def solve_heat_eq(u_0):
            u = u_0
            for _ in range(self._nb_iterations):
                laplacian_u = tf.nn.conv2d(u, self._laplacian_kernel, strides=[1,1,1,1], padding="SAME")
                u = tf.clip_by_value(u + laplacian_u + u_0, 0.0, 1.0)
            return u

        u = solve_heat_eq(u_ori)
        nuage = tfa.image.gaussian_filter2d(
            image = tf.clip_by_value(tf.random.normal((32, 32, 1), mean=0.5, stddev=0.6), 0.0, 1.0),
            filter_shape = 10,
            sigma = 2.0
        )

        u = tf.image.crop_to_bounding_box(u, self._padding_cst, self._padding_cst, self._heat_equation_size_alone, self._heat_equation_size_alone)
        u = tf.image.resize(u, self._size_tf[:2])
        nuage = tf.image.resize(nuage, self._size_tf[:2])
        bg = tf.cast(tf.slice(gt_trimap, [0, 0, 0, 0], [-1, -1, -1, 1]) - u > nuage , dtype="float32")
        fg = tf.cast(tf.slice(gt_trimap, [0, 0, 0, 2], [-1, -1, -1, 1]) - u > nuage , dtype="float32")

        gen_trimap = tf.concat([
            bg,
            1.0 - fg - bg + fg*bg,
            fg
        ], axis=-1)
        
        # Squeeze
        gen_trimap = tf.squeeze(gen_trimap, axis=0)
        gt_trimap = tf.squeeze(gt_trimap, axis=0)
        gt_alpha = tf.squeeze(gt_alpha, axis=0)

        return tf.concat([img, gen_trimap], axis=-1), tf.concat([gt_trimap, gt_alpha], axis=-1)

class LiveComputedDataset:
    def __init__(self, mode, dataset_folder, batch_size=32, img_size = (512, 512), shuffle_buffer = 15000) -> None:
        self._size = img_size
        self._working_res = tf.constant([2*self._size[1], 2*self._size[0], 3])
        self._size_tf = tf.constant([self._size[1], self._size[0], 3+3+3])
        self._dataset_folder = dataset_folder
        self._mode = mode
        self._root_folder = join(self._dataset_folder, self._mode)


        self._laplacian_kernel = tf.expand_dims(tf.expand_dims(
            tf.constant([
                [1,  1, 1],
                [1, -8, 1],
                [1,  1, 1]
            ], dtype="float32")*tf.constant((0.05), dtype="float32"), axis=-1), axis=-1)
        self._heat_equation_size = tf.constant([int(self._size[1]/8), int(self._size[0]/8)], dtype="int32")
        self._heat_equation_size_alone = int(self._size[1]/8)
        self._nb_iterations = 75

        self._padding_cst = 8
        self._central_crop = 1 - self._padding_cst*8/self._size[0]

        self._alpha_folder = tf.constant(join(self._root_folder, "alpha/"))
        self._fg_folder = tf.constant(join(self._root_folder, "fg/"))
        self._bg_folder = tf.constant(join(self._root_folder, "bg/"))

        self._n_test = 60
        self._n_images = 7

        self._batch_size = batch_size
        self._autotune = tf.data.experimental.AUTOTUNE

        self._ds_fg_files = tf.data.Dataset.from_tensor_slices(listdir(join(self._root_folder, "fg/")))
        self._ds_bg_files = tf.data.Dataset.from_tensor_slices(listdir(join(self._root_folder, "bg/")))
        self._ds_val_files = tf.data.Dataset.from_tensor_slices(listdir("/net/homes/r/rseailles/Deep/Alphamatting/alphas/"))

        self._ds_test_fg_files = self._ds_fg_files.take(self._n_test)
        self._ds_test_bg_files = self._ds_bg_files.take(self._n_test).shuffle(shuffle_buffer, reshuffle_each_iteration=True)
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
        bg = tf.image.resize(bg, self._working_res[0:2])

        # Adaptation taille to patch
        # bg = tf.image.random_crop(bg, size=self._working_res)
        gt_alpha = tf.image.resize_with_pad(gt_alpha, self._working_res[0], self._working_res[1])
        fg = tf.image.resize_with_pad(fg, self._working_res[0], self._working_res[1])

        # Position
        limit = tf.math.divide(tf.shape(bg)[:2], 5)
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

        # Expand Alpha
        gt_alpha = tf.expand_dims(tf.slice(gt_alpha, [0,0,0], [-1,-1,1]), axis=0)

        # Ground Truth Trimap
        @tf.function
        def build_gt_trimap(gt_alpha):
            trimap_bg = tf.cast(gt_alpha <= 0.1, dtype="float32")
            trimap_fg = tf.cast(gt_alpha >= 0.95, dtype="float32")
            trimap_uk = 1.0 - trimap_bg - trimap_fg + trimap_bg*trimap_fg
            return tf.concat([trimap_bg, trimap_uk, trimap_fg], axis=-1)

        gt_trimap = build_gt_trimap(gt_alpha)

        # Gen Trimap
        @tf.function
        def build_initial_condition(gt_alpha, strengh=10): # Strengh should be > 8 (because right after this, the resize operation divide the image dimensions by 8
            dilated = gen_nn_ops.max_pool_v2(gt_alpha, [1, strengh, strengh, 1], [1, 1, 1, 1], "SAME")
            eroded = -gen_nn_ops.max_pool_v2(-gt_alpha, [1, strengh, strengh, 1], [1, 1, 1, 1], "SAME")

            trimap_bg = tf.cast(dilated <= 0.1, dtype="float32")
            trimap_fg = tf.cast(eroded >= 0.95, dtype="float32")
            trimap_uk = 1.0 - trimap_bg - trimap_fg + trimap_bg*trimap_fg
            return trimap_uk
        
        u_ori = build_initial_condition(gt_alpha, strengh=10)

        # Resize and pad before use
        u_ori = tf.pad(
            tf.image.resize(u_ori, self._heat_equation_size), 
            paddings=tf.constant(
                [
                    [0,0],
                    [self._padding_cst,self._padding_cst],
                    [self._padding_cst,self._padding_cst],
                    [0,0]
                ]), 
            mode="SYMMETRIC")

        @tf.function
        def solve_heat_eq(u_0):
            u = u_0
            for _ in range(self._nb_iterations):
                laplacian_u = tf.nn.conv2d(u, self._laplacian_kernel, strides=[1,1,1,1], padding="SAME")
                u = tf.clip_by_value(u + laplacian_u + u_0, 0.0, 1.0)
            return u

        u = solve_heat_eq(u_ori)
        nuage = tfa.image.gaussian_filter2d(
            image = tf.clip_by_value(tf.random.normal((16, 16, 1), mean=0.5, stddev=0.6), 0.0, 1.0),
            filter_shape = 5,
            sigma = 1.0
        )

        u = tf.image.crop_to_bounding_box(u, self._padding_cst, self._padding_cst, self._heat_equation_size_alone, self._heat_equation_size_alone)
        u = tf.image.resize(u, self._size_tf[:2])
        nuage = tf.image.resize(nuage, self._size_tf[:2])
        bg = tf.cast(tf.slice(gt_trimap, [0, 0, 0, 0], [-1, -1, -1, 1]) - u > nuage , dtype="float32")
        fg = tf.cast(tf.slice(gt_trimap, [0, 0, 0, 2], [-1, -1, -1, 1]) - u > nuage , dtype="float32")

        gen_trimap = tf.concat([
            bg,
            1.0 - fg - bg + fg*bg,
            fg
        ], axis=-1)
        
        # Squeeze
        gen_trimap = tf.squeeze(gen_trimap, axis=0)
        gt_trimap = tf.squeeze(gt_trimap, axis=0)
        gt_alpha = tf.squeeze(gt_alpha, axis=0)

        return tf.concat([img, gen_trimap], axis=-1), tf.concat([gt_trimap, gt_alpha], axis=-1)


# df = DeepDataset("/net/rnd/DEV/Datasets_DL/alpha_matting/deep38/", batch_size=1, img_size=(32*3, 32*3), size_dividor=32, max_size_factor=2)
# for l in tqdm(df._ds_train_files):
#     _ = df.preprocess(l)

# print("end")
