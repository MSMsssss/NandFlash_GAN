import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from model_DCGAN.main import Generator, Discriminator

g = Generator()
d = Discriminator()
noise = torch.randn(32, 11, 1, 1)
fake = g(noise, 6)
print(fake.shape)
print(d(fake, 6).shape)
