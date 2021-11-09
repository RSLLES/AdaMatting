import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from dataset import DeepDataset
from train_full import FullAdaMatting

##################
### Parameters ###
##################

weights_folder_path = "/net/homes/r/rseailles/Deep/OwnAdaMatting/saves/FullAdaMatting/10-29_19h05"

#############
### Utils ###
#############

def print_metrics(L):
    for name, value in L:
        print(f"{name} -> {value}")

############
### Code ###
############

size = 6
img_size = size*32
batch_size = 5
ds = DeepDataset(
    "/net/rnd/DEV/Datasets_DL/alpha_matting/deep38/", 
    batch_size=batch_size, 
    squared_img_size=img_size,
    max_size_factor=3.0
)

network = FullAdaMatting(
    dataset=ds,
    depth=32,
    n_log_images=5,
    period_test=60*30,
    learning_rate = 0.0001
)

# network.load(weights_folder_path)
# network.save_light_model(weights_folder_path)
network.load_light_model(weights_folder_path)

### 1) Regular Model test
# print("Regular model test : ")
# network.test()
# print_metrics(network.get_test_metrics())

print("Optimized model test : ")
network.test()
print_metrics(network.get_test_metrics())

