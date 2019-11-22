import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from model_DCGAN.main import Generator, Discriminator

s = "generator_epoch_100.pth"
epoch = int(s[s.rfind("_") + 1:s.rfind(".")])
print(epoch)