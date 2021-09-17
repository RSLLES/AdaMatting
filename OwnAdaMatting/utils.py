import matplotlib
import io

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import keras_model_summary

def generate_graph(tf_writer, model):
    with tf_writer.as_default():
      with tf.summary.record_if(True):
        summary_writable = (
            model._is_graph_network or  # pylint: disable=protected-access
            model.__class__.__name__ == 'Sequential')  # pylint: disable=protected-access
        if summary_writable:
            keras_model_summary('keras', model, step=0)
        else:
            print("Can't write graph")


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


def image_grid(df, model, n):
    nb_categories = 4

    fig, axs = plt.subplots(n, nb_categories)
    scale_img = 5
    fig.set_size_inches(nb_categories*scale_img,n*scale_img) 
    
    for row, data in zip(range(n), df):
        x, y = data
        out = model.predict(x)
        for i, title, d in zip(
                                range(nb_categories), 
                                ["Patched Image", "User's trimap input", "Refined Trimap", "Ground Truth Trimap"],
                                [x[0, :,:,0:3], x[0, :,:,3:6], out[0, :,:,:], y[0, :,:,:]]
                            ):
            axs[row, i].imshow(d)
            axs[row, i].axis("off")
            if row == 0:
                axs[row, i].set_title(title)

    return fig