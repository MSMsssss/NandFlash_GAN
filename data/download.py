import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
sys.path.append(root_path + "\\data")

from data.dataset import Dataset, SqlConfig
import torch.utils.data
import numpy as np
import torch

data_set = Dataset()
data_loader = torch.utils.data.DataLoader(data_set, batch_size=len(data_set), shuffle=False)

for data, condition in data_loader:
    np.save(root_path + "/data/download_data/data.npy", data.numpy())
    np.save(root_path + "/data/download_data/condition.npy", condition.numpy())

print("下载完成")
