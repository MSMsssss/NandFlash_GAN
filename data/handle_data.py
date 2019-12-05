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

file_prefix = "pe_data"
data_root_path = cur_path + "/all_data/"

with open(data_root_path + "import.log") as f:
    file_list = f.read().split("\n")
    file_list.remove("")


def run():
    array_list = []
    for file in file_list:
        array_list.append(np.load(data_root_path + file_prefix + "_" + file + ".npy"))
        print("%s load success" % file)
    rtn = np.concatenate(array_list, axis=0)
    np.save(data_root_path + "cat_data/all_%s.npy" % file_prefix, rtn.astype(np.float32))
    print(rtn.shape)


if __name__ == '__main__':
    run()
