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


if __name__ == "__main__":
    data_path = "e:/nandflash_data/2019_8_9/"
    l = []
    with open(data_path + "002.log") as f:
        for i, line in enumerate(f):
            if i > 10:
                break
            l.append(line)
            print(l)
