import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from model_DCGAN.main import Generator, Discriminator


t = torch.ones((2, 3))
print(t)
t = t * 2
print(t)
t.fill_(1.0)
print(t)