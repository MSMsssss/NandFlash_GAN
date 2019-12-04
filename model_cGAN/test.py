import os
import sys
import time

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

a = 10


def f():
    global a
    a += 1


if __name__ == "__main__":
    s = "[20000   100 C0 L0  B2    P0    SLC   ][  78][       1][0      0      1      0      0      0      0      0   " \
        "   0      0      0      0      0      0      0      0      ]\n"
    temp = s[1:-2].split("][")
    for i in range(len(temp)):
        temp[i] = temp[i].split(" ")
        for _ in range(temp[i].count("")):
            temp[i].remove("")

        print(temp[i])
