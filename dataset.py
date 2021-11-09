import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.ops import gen_nn_ops

from os.path import join
from os.path import isfile  

class DeepDataset:
    def __init__(self, dataset_folder, batch_size=32, squared_img_size = 320, max_size_factor=2.0, shuffle_buffer = 150000) -> None:
        """
        Parameters
        ----------
        dataset_folder : str
            Root folder of the dataset. It must contains at least one of train.csv or test.csv
        batch_size : int, optional
            batch size to use
            default = 32
        squared_size_image : int, optional
            size of the image to build. It should be a factor of size_dividor
            default = 320
        max_size_factor : float, optional
            To add som randomness to the process, it could be good to work entirely with a different image sizes and
            ultimately resize the image to squared_size_image.
            This parameter set the max coefficient which can be used to work with an larger size.
            default = 2.0
        shuffle_buffer : int, optional
            Tensorflow shuffle buffer.
            default = 150 000
        """


        ### Assert 
        # - datasets must exists
        assert isfile(join(dataset_folder, "train.csv")) or isfile(join(dataset_folder, "test.csv"))
        
        ### Network constants
        self.batch_size = batch_size

        ### Around Size
        self.size = squared_img_size
        self.size_tf = tf.constant([self.size, self.size])
        self.max_size_factor = int(round(max_size_factor))
        self.nb_possible_sizes = 100
        self.all_croping_sizes = tf.repeat(
            tf.cast(tf.linspace(self.size, int(self.size*self.max_size_factor), self.nb_possible_sizes, 1), dtype="int32"),
            repeats=2,
            axis=-1
        )
        self.shift_proba_cst = 5
        self.sizes_proba = tf.expand_dims(tf.constant([1/k for k in range(1 + self.shift_proba_cst ,self.nb_possible_sizes + 1 + self.shift_proba_cst)], dtype="float32"), axis=0)

        ### Around Heat Equation
        self.diffusivity_constant = tf.constant((0.04), dtype="float32")
        self.laplacian_kernel = self.diffusivity_constant*tf.expand_dims(tf.expand_dims(
            tf.constant([
                [1,  1, 1],
                [1, -8, 1],
                [1,  1, 1]
            ], dtype="float32"), axis=-1), axis=-1)
        self.heat_equation_solving_size = int(self.size/4)
        self.heat_equation_solving_size_tf = tf.constant([self.heat_equation_solving_size, self.heat_equation_solving_size], dtype="int32")
        self.nb_iterations = 30
        self.padding_cst = 8
        self.padding_window = tf.constant([self.padding_cst, self.padding_cst])


        ### Load datasets
        self.columns = ["fg", "alpha", "bg"]
        self.autotune = tf.data.experimental.AUTOTUNE
        self.dataset_folder = dataset_folder
        self.train_csv = join(self.dataset_folder, "train.csv")
        self.test_csv = join(self.dataset_folder, "test.csv")

        if isfile(join(dataset_folder, "train.csv")):
            self.ds_train_files = tf.data.experimental.make_csv_dataset(
                self.train_csv,
                column_names=self.columns,
                batch_size=1,
                header=False,
                num_epochs=1).shuffle(shuffle_buffer, reshuffle_each_iteration=True)
            self.ds_train = self.ds_train_files.map(lambda x : self.preprocess(x), num_parallel_calls = self.autotune).batch(batch_size).prefetch(self.autotune)

        if isfile(join(dataset_folder, "test.csv")):
            self.ds_test_files = tf.data.experimental.make_csv_dataset(
                self.test_csv, 
                column_names=self.columns,
                batch_size=1,
                header=False,
                num_epochs=1)
            self.ds_test = self.ds_test_files.map(lambda x : self.preprocess(x), num_parallel_calls = self.autotune).batch(batch_size).prefetch(self.autotune)

    @tf.function
    def preprocess(self, line) -> tf.Tensor:
        """
        Parameters
        ----------
        line : strings tf.Tensor, shape = (3)
        Contains :
            - line["fg"] : Path to foreground image
            - line["alpha"] : Path to the alpha of the previously given foreground image
            - line["bg] : background image to use

        Return
        ------
        - x : tf.tensor of shape (img_size, img_size, 6) which is the concatenation on the last axis of :
            - input img of shape (img_size, img_size, 3)
            - generate input trimap of shape (img_size, img_size, 3)
        - y : tf.tensor of shape (img_size, img_size, 4) which is the concatenation on the last axis of :
            - ground truth trimap of shape (img_size, img_size, 3)
            - ground truth alpha of shape (img_size, img_size, 1)
        """

        ### Importation

        gt_alpha = DeepDataset.import_img(line, "alpha")
        fg = DeepDataset.import_img(line, "fg")
        bg = DeepDataset.import_img(line, "bg")
        gt_alpha = tf.slice(gt_alpha, [0,0,0], [-1, -1, 1]) # Safety

        ### Angle
        # angle = tf.random.uniform(shape=[], minval=-0.1*3.1416, maxval=0.1*3.1416)
        # gt_alpha = tfa.image.rotate(gt_alpha, angle)
        # fg = tfa.image.rotate(fg, angle)

        ### Flip
        fg_and_alpha = tf.concat([fg, gt_alpha], axis=-1)
        fg_and_alpha = tf.image.random_flip_left_right(fg_and_alpha)
        # fg_and_alpha = tf.image.random_flip_up_down(fg_and_alpha)
        fg = tf.slice(fg_and_alpha, [0,0,0], [-1,-1,3])
        gt_alpha = tf.slice(fg_and_alpha, [0,0,3], [-1,-1,1])

        ### Crop
        # 1) Pick a random pixel in the unknown zone
        unknown_zone = DeepDataset.extract_unknown_zone_HWC(gt_alpha)
        # 2D -> Flatten
        unknown_zone = tf.reshape(unknown_zone, [1, tf.shape(unknown_zone)[0]*tf.shape(unknown_zone)[1]])
        # Pick a pixel
        rand_pixel_flatten = tf.squeeze(tf.cast(tf.random.categorical(tf.math.log(unknown_zone), 1), dtype="int32"))
        # Flatten -> 2D
        rand_pixel = tf.stack([
            tf.cast(tf.math.divide(rand_pixel_flatten, tf.shape(gt_alpha)[1]), dtype="int32"),
            tf.math.floormod(rand_pixel_flatten, tf.shape(gt_alpha)[1])
            ])
        # Clip with some room for safety
        rand_pixel = tf.clip_by_value(
            rand_pixel, 
            clip_value_min = self.padding_window, 
            clip_value_max = tf.shape(gt_alpha)[:2] - self.padding_window)

        # Pick a random size to work with
        croping_size = self.all_croping_sizes[
            tf.squeeze(tf.random.categorical(tf.math.log(self.sizes_proba), 1))
        ]
        croping_size = tf.repeat(croping_size, repeats=2)

        # Croping
        fg = DeepDataset.safely_crop_around_one_pixel(fg, croping_size, rand_pixel)
        gt_alpha = DeepDataset.safely_crop_around_one_pixel(gt_alpha, croping_size, rand_pixel)
        bg = tf.image.resize(bg, croping_size)
        
        ### 1) Build input img
        img = fg*gt_alpha + bg*(1.0-gt_alpha)
        img = DeepDataset.data_augmentation(img)
        img = tf.image.resize(img, self.size_tf)

        ### 2) Build ground truth alpha
        gt_alpha = tf.image.resize(gt_alpha, self.size_tf)

        ### 3) Build ground truth trimap
        gt_trimap = DeepDataset.build_gt_trimap(gt_alpha)

        ### 4) Build input trimap 
        gen_trimap = self.generate_input_trimap(gt_alpha, gt_trimap)

        return tf.concat([img, gen_trimap], axis=-1), tf.concat([gt_trimap, gt_alpha], axis=-1)


    @tf.function
    def generate_input_trimap(self, alpha, gt_trimap):
        # Those algorithms are working with tensor of rank 4 (NHWC format)
        alpha = tf.expand_dims(alpha, axis=0)
        gt_trimap = tf.expand_dims(gt_trimap, axis=0)

        u_ori = DeepDataset.extract_unknown_zone_NHWC(alpha, pre_dilation_strengh=10)
        u_ori = tf.image.resize(u_ori, self.heat_equation_solving_size_tf)
        

        # Solving
        # Padding to avoid indesirable border effects
        u_ori = self.pad(u_ori)
        u = self.solve_heat_eq(u_ori)
        u = self.unpad(u)

        # Randomize
        random_map = self.get_random_gaussian_map()
        random_map = tf.expand_dims(random_map, axis=0)

        # Resize
        u = tf.image.resize(u, self.size_tf)
        random_map = tf.image.resize(random_map, self.size_tf)

        # Build trimap
        bg = tf.cast(tf.slice(gt_trimap, [0, 0, 0, 0], [-1, -1, -1, 1]) - u > random_map , dtype="float32")
        fg = tf.cast(tf.slice(gt_trimap, [0, 0, 0, 2], [-1, -1, -1, 1]) - u > random_map , dtype="float32")

        gen_trimap = tf.concat([
            bg,
            1.0 - fg - bg + fg*bg,
            fg
        ], axis=-1)

        return tf.squeeze(gen_trimap, axis=0)



#########################
### Utility functions ###
#########################

    @tf.function
    def import_img(line, type_img):
        return tf.image.convert_image_dtype(tf.image.decode_png(tf.io.read_file(tf.squeeze(line[type_img]))), dtype="float32")

    @tf.function
    def smoothed_clamp(x, mu, eps = 0.01):
        if x - mu < - eps:
            return 0.0
        if x - mu > eps:
            return 1.0
        return (x-mu)/(2*eps) + 0.5

    @tf.function
    def build_gt_trimap(alpha, pre_dilation_strengh=0):
        if pre_dilation_strengh > 0:
            dilated = gen_nn_ops.max_pool_v2(alpha, [1, pre_dilation_strengh, pre_dilation_strengh, 1], [1, 1, 1, 1], "SAME")
            eroded = -gen_nn_ops.max_pool_v2(-alpha, [1, pre_dilation_strengh, pre_dilation_strengh, 1], [1, 1, 1, 1], "SAME")
        else:
            dilated = alpha
            eroded = alpha
        # trimap_bg = tf.cast(dilated <= 0.01, dtype="float32")
        trimap_bg = 1.0 - DeepDataset.smoothed_clamp(dilated, 0.01)
        # trimap_fg = tf.cast(eroded >= 0.94, dtype="float32")
        trimap_fg = DeepDataset.smoothed_clamp(eroded, 0.94)
        # trimap_uk = 1.0 - trimap_bg - trimap_fg + trimap_bg*trimap_fg
        trimap_uk = 1.0 - trimap_bg - trimap_fg
        return tf.concat([trimap_bg, trimap_uk, trimap_fg], axis=-1)

    @tf.function
    def extract_unknown_zone_NHWC(alpha, pre_dilation_strengh=0):
        return tf.slice(
            DeepDataset.build_gt_trimap(alpha, pre_dilation_strengh=0),
            [0, 0, 0, 1], [-1, -1, -1, 1]
            )

    @tf.function
    def extract_unknown_zone_HWC(alpha, pre_dilation_strengh=0):
        return tf.slice(
            DeepDataset.build_gt_trimap(alpha, pre_dilation_strengh=0),
            [0, 0, 1], [-1, -1, 1]
            )

    @tf.function
    def safely_crop_around_one_pixel(img, size, pixel):
        # assert tf.rank(pixel) == 1 and tf.rank(size) == 1
        # assert tf.shape(pixel)[0] == 2 and tf.shape(size)[0] == 2

        half = tf.cast(tf.math.divide(size, 2), dtype="int32") + 1 # +1 To avoid rounding errors
        d_hg = -tf.clip_by_value(pixel - half, -10000, 0)
        d_bd = -tf.clip_by_value(tf.shape(img)[:2] - pixel - half, -10000, 0)

        img = tf.image.pad_to_bounding_box(
            img, 
            d_hg[0], 
            d_hg[1], 
            tf.shape(img)[0] + d_bd[0] + d_hg[0], 
            tf.shape(img)[1] + d_bd[1] + d_hg[1])

        img = tf.image.crop_to_bounding_box(
            img, 
            pixel[0] + d_hg[0] - half[0], 
            pixel[1] + d_hg[1] - half[1], 
            size[0], 
            size[1])

        return img

    @tf.function
    def data_augmentation(img):
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_saturation(img, 0.7, 1.5)
        # img = tf.image.random_hue(img, 0.1)

        return tf.clip_by_value(img, 0.0, 1.0)

    @tf.function
    def pad(self, img):
        return tf.pad(
            img, 
            paddings=tf.stack(
                [
                    tf.constant([0,0], dtype="int32"),
                    self.padding_window,
                    self.padding_window,
                    tf.constant([0,0], dtype="int32")
                ], axis=0), 
            mode="SYMMETRIC")

    @tf.function
    def unpad(self, img):
        return tf.image.crop_to_bounding_box(
            img, 
            self.padding_cst, 
            self.padding_cst, 
            self.heat_equation_solving_size, 
            self.heat_equation_solving_size)


    @tf.function
    def solve_heat_eq(self, u_0):
        u = u_0
        for _ in range(self.nb_iterations):
            laplacian_u = tf.nn.conv2d(u, self.laplacian_kernel, strides=[1,1,1,1], padding="SAME")
            u = tf.clip_by_value(u + laplacian_u + u_0, 0.0, 1.0)
        return u

    @tf.function
    def get_random_gaussian_map(self):
        return tfa.image.gaussian_filter2d(
            image = tf.clip_by_value(
                tf.random.normal((self.heat_equation_solving_size, self.heat_equation_solving_size, 1), 
                mean=0.5, stddev=0.6), 
                0.0, 1.0),
            filter_shape = 10,
            sigma = 2.0
        )


#############
### Tests ###
#############

if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    ds = DeepDataset("/net/rnd/DEV/Datasets_DL/alpha_matting/deep38/", batch_size=2, squared_img_size=160, max_size_factor=3.0)
    ds.preprocess(next(iter(ds.ds_test_files)))
    print("Done")