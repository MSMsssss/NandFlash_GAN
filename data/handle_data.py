import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(cur_path)
sys.path.append(root_path)

import torch.utils.data
import numpy as np
import torch
import argparse


# 合并所有数据
def merge_data():
    file_prefix = "pe_data"
    data_root_path = cur_path + "/all_data/"

    with open(data_root_path + "import.log") as f:
        file_list = f.read().split("\n")
        file_list.remove("")

    result = np.load(data_root_path + file_prefix + "_" + file_list[0] + ".npy").astype(np.float32)
    print("%s load success" % file_list[0])
    file_list.remove(file_list[0])

    for file in file_list:
        result = np.concatenate((result,
                                 np.load(data_root_path + file_prefix + "_" + file + ".npy").astype(np.float32)),
                                axis=0)
        print("%s load success" % file)

    np.save(data_root_path + "cat_data/all_%s.npy" % file_prefix, result)
    print(result.shape)


if __name__ == '__main__':
    merge_data()
