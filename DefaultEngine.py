import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from datetime import datetime
from time import time
from os.path import join

from Engine import Engine

class DefaultEngine(Engine):

    def __init__(self, dataset, name, period_test, lr) -> None:
        super().__init__(dataset)
        
        date = datetime.now().strftime("%m-%d_%Hh%M")
        self.save_dir = f"OwnAdaMatting/saves/{name}/{date}/"
        self.log_dir = f"OwnAdaMatting/logs/{name}/{date}/"
        self.train_writer = tf.summary.create_file_writer(join(self.log_dir, f"train/"))
        self.test_writer = tf.summary.create_file_writer(join(self.log_dir, f"test/"))
        self.period_test = period_test
        self.optimizer = Adam(learning_rate=lr)


    def main_loop_condition(self):
        return True

    def training_loop_condition(self):
        return True

    def testing_condition(self):
        return time() - self.last_test_time > self.period_test
    
    def reset_testing_condition(self):
        self.last_test_time = time()

    def saving_condition(self):
        return self.testing_condition()

    def reset_saving_condition(self): 
        pass

    def training_log_condition(self):
        return True

    def reset_training_log_condition(self):
        pass

    def testing_log_condition(self):
        return True

    def reset_testing_log_condition(self):
        pass
