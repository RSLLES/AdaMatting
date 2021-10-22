import tensorflow as tf
import tensorflow as tf

from DefaultEngine import DefaultEngine
from loss import MultiTaskLoss, AlphaLoss, AdaptiveTrimapLoss
from dataset import DeepDataset
from network import get_model

class FullAdaMatting(DefaultEngine):
    def __init__(self, input_shape, dataset, depth, n_log_images, period_test, learning_rate) -> None:
        """
        Train the complete version of the AdaMatting network
        Parameters :
            - input_shape (int, int):
            Size of the image

            - dataset (DeepDataset)
            Dataset to use

            - depth (int)
            Depth of the network

            - n_log_images (int)
            How many images should be output when testing

            - period_test (float)
            Duration between two tests

            - learning_rate (float)
            Learning rate to use with Adam

        """ 

        super().__init__(
            dataset=dataset,
            name=type(self).__name__,
            period_test=period_test,
            lr=learning_rate
        )

        self.model, _, _, _ = get_model(depth=depth, input_shape=input_shape)
        self.n_log_images = n_log_images

        self.loss_alpha_func = AlphaLoss()
        self.loss_trimap_func = AdaptiveTrimapLoss()
        self.loss_multitask_func = MultiTaskLoss()


    def compute_losses(self, y_true, y_pred):
        loss_alpha = self.loss_alpha_func(y_true, y_pred)
        loss_trimap = self.loss_trimap_func(y_true, y_pred)
        loss = self.loss_multitask_func(y_true, y_pred, loss_trimap, loss_alpha)
        return loss, loss_alpha, loss_trimap

    def extract_training_loss(self, losses):
        return losses[0]

    def update_train_metrics(self, losses):
        self.Loss_train.append(losses[0])
        self.Loss_alpha_train.append(losses[1])
        self.Loss_trimap_train.append(losses[2])
        self.S1.append(tf.exp(0.5*self.model.layers[-1].kernel[0]).numpy()[0])
        self.S2.append(tf.exp(0.5*self.model.layers[-1].kernel[1]).numpy()[0])

    def get_train_metrics(self):
        mean = lambda L : sum(L)/len(L) if len(L) > 0 else -1
        return [
            ("Loss", mean(self.Loss_train)),
            ("AlphaLoss", mean(self.Loss_alpha_train)),
            ("TrimapLoss", mean(self.Loss_trimap_train)),
            ("S1", mean(self.S1)),
            ("S2", mean(self.S2))
        ]

    def reset_train_metrics(self):
        self.Loss_train = []
        self.Loss_alpha_train = []
        self.Loss_trimap_train = []
        self.S1 = []
        self.S2 = []


    def get_test_metrics(self):
        mean = lambda L : sum(L)/len(L) if len(L) > 0 else -1
        return [
            ("Loss", mean(self.Loss_test)),
            ("AlphaLoss", mean(self.Loss_alpha_test)),
            ("TrimapLoss", mean(self.Loss_trimap_test)),
        ]

    def reset_test_metrics(self):
        self.Loss_test = []
        self.Loss_alpha_test = []
        self.Loss_trimap_test = []

    def update_test_metrics(self, losses):
        self.Loss_test.append(losses[0])
        self.Loss_alpha_test.append(losses[1])
        self.Loss_trimap_test.append(losses[2])

    def get_columns(self):
        return [
            "Patched Image", 
            "User's trimap input", 
            "Refined Trimap", 
            "Ground Truth Trimap", 
            "Refined Alpha", 
            "Ground Truth Alpha",
            "Composed"
        ]

    def get_grid(self):
        grid =  []
        for i, data in zip(range(self.n_log_images), self.ds.ds_test):
            x_batch, y_batch = data
            x_batch = tf.slice(x_batch, [0,0,0,0],[1,-1,-1,-1])
            y_batch = tf.slice(y_batch, [0,0,0,0],[1,-1,-1,-1])
            out_alpha, out_trimap, _ = self.model(x_batch, training=False)

            img = tf.squeeze(tf.slice(x_batch, [0,0,0,0], [1, -1, -1, 3]), axis=0)
            in_trimap = tf.squeeze(tf.slice(x_batch, [0,0,0,3], [1, -1, -1, 3]), axis=0)
            gt_trimap = tf.squeeze(tf.slice(y_batch, [0,0,0,0], [1, -1, -1, 3]), axis=0)
            gt_alpha = tf.squeeze(tf.slice(y_batch, [0,0,0,3], [1, -1, -1, 1]), axis=0)
            out_trimap = tf.squeeze(out_trimap, axis=0)
            out_alpha = tf.squeeze(out_alpha, axis=0)

            gt_alpha = tf.repeat(tf.clip_by_values(gt_alpha, 0.0, 1.0), repeats=3, axis=-1)
            out_alpha = tf.repeat(tf.clip_by_values(out_alpha, 0.0, 1.0), repeats=3, axis=-1)

            background = tf.concat([
                tf.zeros(shape=out_alpha.shape),
                tf.ones(shape=out_alpha.shape),
                tf.zeros(shape=out_alpha.shape)
            ], axis=-1)

            composed = img*out_alpha + background*(1.0 - out_alpha)

            grid.append([img, in_trimap, out_trimap, gt_trimap, out_alpha, gt_alpha, composed])
        return grid



if __name__ == "__main__":
    size = 5
    img_size = size*32
    batch_size = 5
    ds = DeepDataset(
        "/net/rnd/DEV/Datasets_DL/alpha_matting/deep38/", 
        batch_size=batch_size, 
        squared_img_size=img_size,
        max_size_factor=3.0
    )

    network = FullAdaMatting(
        input_shape=(img_size, img_size),
        dataset=ds,
        depth=32,
        n_log_images=5,
        period_test=60*30,
        learning_rate = 0.0001
    )
    # network.load("OwnAdaMatting/saves/10-22_12h46/FullAdaMatting")
    network.train()
        



