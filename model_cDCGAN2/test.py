import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)
sys.path.append(cur_path)

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

err_data = np.load(cur_path + "/download_data/data.npy")
pe_data = np.load(cur_path + '/download_data/condition.npy')

print(err_data.shape, pe_data.shape)