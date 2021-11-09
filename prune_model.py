import tensorflow as tf
import tensorflow_model_optimization as tfmot

from train_full import FullAdaMatting
from loss import MultiTaskLoss, AlphaLoss, AdaptiveTrimapLoss
from dataset import DeepDataset
from network import get_model

class PrunedAdaMatting(FullAdaMatting):
    def __init__(self, dataset, initial_weights_path, depth, n_log_images, period_test, learning_rate) -> None:
        """
        Tries to pruned the model to make it faster

        """ 

        super().__init__(
            dataset, depth, n_log_images, period_test, learning_rate
        )
        self.load(initial_weights_path)
        self.model = tfmot.sparsity.keras.prune_low_magnitude(self.model)
        self.model.optimizer = self.optimizer
        self.step_callback = tfmot.sparsity.keras.UpdatePruningStep()
        self.step_callback.set_model(self.model)

    def callback_begin_training(self):
        self.step_callback.on_train_begin()

    def callback_before_train_one_batch(self):
        self.step_callback.on_train_batch_begin(batch=-1)

    def callback_after_train_one_batch(self):
        self.step_callback.on_epoch_end(batch=-1)

    def update_train_metrics(self, losses):
        self.Loss_train.append(losses[0])
        self.Loss_alpha_train.append(losses[1])
        self.Loss_trimap_train.append(losses[2])
        
    def get_train_metrics(self):
        mean = lambda L : sum(L)/len(L) if len(L) > 0 else -1
        return [
            ("Loss", mean(self.Loss_train)),
            ("AlphaLoss", mean(self.Loss_alpha_train)),
            ("TrimapLoss", mean(self.Loss_trimap_train)),
        ]

    def reset_train_metrics(self):
        self.Loss_train = []
        self.Loss_alpha_train = []
        self.Loss_trimap_train = []    

    def save(self):
        self.model_pruned = self.model
        self.model = tfmot.sparsity.keras.strip_pruning(self.model_pruned)
        super().save()
        self.model = self.model_pruned


if __name__ == "__main__":
    size = 6
    img_size = size*32
    batch_size = 5
    ds = DeepDataset(
        "/net/rnd/DEV/Datasets_DL/alpha_matting/deep38/", 
        batch_size=batch_size, 
        squared_img_size=img_size,
        max_size_factor=3.0
    )

    network = PrunedAdaMatting(
        dataset=ds,
        initial_weights_path = "/net/homes/r/rseailles/Deep/OwnAdaMatting/saves/FullAdaMatting/10-29_19h05",
        depth=32,
        n_log_images=5,
        period_test=60*15,
        learning_rate = 0.0001
    )
    network.train()
        



