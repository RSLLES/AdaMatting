import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from network import get_model

def run(
            img_path, tri_path, out_path,
            model, depth
        ):
    def open_img (path):
        img = tf.image.convert_image_dtype(tf.image.decode_image(tf.io.read_file(path)), dtype="float32")
        x,y = int(img.shape[0]/(depth))*depth, int(img.shape[1]/(depth))*depth
        img = tf.image.resize(img, (x, y))
        return img

    img = open_img(img_path)
    tri = tf.slice(open_img(tri_path), [0,0,0], [-1, -1, 1])

    def extract_trimap(a):
        return tf.concat([
            tf.cast(a <= 0.01, dtype=tf.float32),
            tf.cast((a > 0.01) & (a < 0.99), dtype=tf.float32),
            tf.cast(a >= 0.99, dtype=tf.float32)
        ], axis=-1)

    tri = extract_trimap(tri)

    x = tf.concat([img, tri], axis=-1)
    y = tf.squeeze(model(tf.expand_dims(x, axis=0))[1], axis=0)
    y = tf.clip_by_value(y, 0.0, 1.0)

    alpha = tf.repeat(y, 3, axis=-1)

    plt.imsave(out_path, alpha.numpy())


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage : {sys.argv[0]} [img_path] [trimap_path] [output_path]")
        exit()

    depth=32
    path_weights = "/net/homes/r/rseailles/Deep/OwnAdaMatting/saves/FullAdaMatting/10-29_19h05/11-04_17h46.h5"
    model = get_model(depth=depth)[0]
    model.load_weights(path_weights)
    run(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        model,
        depth
    )