import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(cur_path)
sys.path.append(root_path)

from data.dataset import Dataset, SqlConfig
import torch.utils.data
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--err_file_name", default="data.npy", help="块错误信息文件名")
parser.add_argument("--condition_file_name", default="condition.npy", help="条件信息文件名")
opt = parser.parse_args()


def download_err_data():
    data_set = Dataset()
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=len(data_set), shuffle=False)

    for data, condition in data_loader:
        np.save(root_path + "/data/download_data/" + opt.err_file_name, data.numpy())
        np.save(root_path + "/data/download_data/condition.npy" + opt.condition_file_name, condition.numpy())

    print("下载完成")


if __name__ == "__main__":
    download_err_data()