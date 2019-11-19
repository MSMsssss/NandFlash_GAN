import torch
import torch.nn as nn
import numpy as np
import os
import sys

w = torch.tensor([1., 2., 3.], requires_grad=True)
x = torch.tensor([1., 5., 4.], requires_grad=False)
y = w * x
y.backward(torch.tensor([1., 1., 1.]))
print(w.grad)