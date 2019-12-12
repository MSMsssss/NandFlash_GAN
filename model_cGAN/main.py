import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)
sys.path.append(cur_path)

import argparse
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from utils.utils import random_int_with_sum, norm_range
from model_cGAN.config import Config

config = Config()
parser = argparse.ArgumentParser()

# train与eval共用参数

# train参数

# eval参数

opt = parser.parse_args()
print(opt)


def merge_data(num_array, probability_array):
    for i in probability_array.shape[0]:
        probability_array[i] = norm_range(probability_array[i]) / probability_array[i].sum()

    temp_data = np.zeros(shape=probability_array.shape, dtype=np.int32)

    for i in probability_array.shape[0]:
        temp_data[i] = (probability_array[i] * num_array[i] + 0.5).astype(np.int32)

    fake_data = np.zeros(shape=(num_array.shape[0], 2304, 16), dtype=np.int32)
    for i in range(num_array.shape[0]):
        for j in range(2304):
            fake_data[i][j] = random_int_with_sum(temp_data[i][j], 16)

    return fake_data


def run():
    num_array = np.load("xxx.npy")
    probability_array = np.load("xxx.npy")
    fake_data = merge_data(num_array, probability_array)
    print(fake_data)


if __name__ == "__main__":
    a = np.array([1.7,2.2,3.4], dtype=np.float32)
    print(a.astype(np.int32))
