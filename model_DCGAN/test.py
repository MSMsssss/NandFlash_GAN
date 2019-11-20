import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from model_DCGAN.main import Generator, Discriminator

g = Generator()
d = Discriminator()
noise = torch.randn(32, 10, 1, 1)
condition = torch.randn(32, 1)
print(condition)
print(condition.view(-1, 1, 1, 1))
# fake = g(noise, condition)
# print(fake.shape)
# print(d(fake, condition).shape)
