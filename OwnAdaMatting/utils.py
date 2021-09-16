import matplotlib
import io

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf

# Stolen from tensorflow official guide:
# https://www.tensorflow.org/tensorboard/image_summaries
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(data, model, n=4):
    nb_categories = 4

    x, y = data
    fig, axs = plt.subplots(n, nb_categories)
    scale_img = 3
    fig.set_size_inches(nb_categories*scale_img,n*scale_img) 
    out = model.predict(x)

    for row in range(n):
        for i, title, d in zip(
                                range(nb_categories), 
                                ["Patched Image", "User's trimap input", "Refined Trimap", "Ground Truth Trimap"],
                                [x[row, :,:,0:3], x[row, :,:,3:6], out[row, :,:,:], y[row, :,:,:]]):
            axs[row, i].imshow(d)
            axs[row, i].axis("off")
            if row == 0:
                axs[row, i].set_title(title)

    return fig