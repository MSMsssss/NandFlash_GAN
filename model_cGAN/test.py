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


a = 0.1 * torch.randn((2000, ))
d = a.histc(20)
left = a.min()
right = a.max()
x = [left + i * (right - left) / d.shape[0] for i in range(d.shape[0])]
plt.close()
plt.plot(x, list(d))
plt.show()

