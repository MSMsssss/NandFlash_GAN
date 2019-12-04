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
import torch.utils.data
from model_cGAN.config import Config
from model_cGAN.dataset import TotalErrDataset, TestDataSet
from data.connect_database import Connect, SqlConfig

with open('e:/nandflash_data/2019_8_1/000.log', "r") as f:
    rtn = f.readlines()


