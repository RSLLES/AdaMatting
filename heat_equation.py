import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.python.ops import gen_nn_ops

alpha_path = "/net/rnd/DEV/Datasets_DL/alpha_matting/picky2/alpha/7435788566250265.jpg"

def open_img (folder):
    img = tf.image.convert_image_dtype(tf.image.decode_image(tf.io.read_file(folder)), dtype="float32")
    x,y = int(img.shape[0]/(16))*16, int(img.shape[1]/(16))*16
    img = tf.image.resize(img, (x, y))
    return img

@tf.function
def constant_dilate(dilated, eroded, strengh=1):
    dilated = gen_nn_ops.max_pool_v2(dilated, [1, strengh, strengh, 1], [1, 1, 1, 1], "SAME")
    eroded = -gen_nn_ops.max_pool_v2(-eroded, [1, strengh, strengh, 1], [1, 1, 1, 1], "SAME")
    return dilated, eroded

@tf.function
def original_trimap(gt_alpha):
    dilated, eroded = constant_dilate(gt_alpha, gt_alpha)

    trimap_bg = tf.cast(dilated <= 0.1, dtype="float32")
    trimap_fg = tf.cast(eroded >= 0.95, dtype="float32")
    trimap_uk = 1.0 - trimap_bg - trimap_fg + trimap_bg*trimap_fg
    return tf.concat([trimap_bg, trimap_uk, trimap_fg], axis=-1)

alpha = tf.expand_dims(tf.slice(open_img(alpha_path), [0,0,0], [-1,-1,1]), axis=0)
trimap_in = original_trimap(alpha)

# Heat Equation
laplacian_kernel = tf.constant([
    [1,  1, 1],
    [1, -8, 1],
    [1,  1, 1]
], dtype="float32")
D = tf.constant((0.05), dtype="float32")

laplacian_kernel = tf.expand_dims(tf.expand_dims(laplacian_kernel, axis=-1), axis=-1)

u_ori = tf.slice(trimap_in, [0, 0, 0, 1], [-1, -1, -1, 1])
u_ori = tf.pad(tf.image.resize(u_ori, (128, 128)), paddings=tf.constant([[0,0],[8,8],[8,8],[0,0]]), mode="SYMMETRIC")
u = u_ori
for _ in range(300):
    laplacian_u = tf.nn.conv2d(u, laplacian_kernel, strides=[1,1,1,1], padding="SAME")
    u = tf.clip_by_value(u + D*laplacian_u + u_ori, 0.0, 1.0)

nuage = tfa.image.gaussian_filter2d(
    image = tf.clip_by_value(tf.random.normal((16, 16, 1), mean=0.5, stddev=0.3), 0.0, 1.0),
    filter_shape = 5,
    sigma = 1.0
    )

u = tf.image.central_crop(u, 8/9)
u = tf.image.resize(u, alpha.shape[1:3])
nuage = tf.image.resize(nuage, alpha.shape[1:3])
bg = tf.cast(tf.slice(trimap_in, [0, 0, 0, 0], [-1, -1, -1, 1]) - u > nuage , dtype="float32")
fg = tf.cast(tf.slice(trimap_in, [0, 0, 0, 2], [-1, -1, -1, 1]) - u > nuage , dtype="float32")

trimap_out = tf.concat([
    bg,
    1.0 - fg - bg + fg*bg,
    fg
], axis=-1)

# Display
fig, axis = plt.subplots(1,4, figsize=(5,5))
axis[0].imshow(tf.squeeze(trimap_in, axis=0))
axis[1].imshow(tf.squeeze(u, axis=0))
axis[2].imshow(tf.squeeze(trimap_out, axis=0))
axis[3].imshow(nuage)
plt.show()

