import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cur_path = root_path + "/data"

import numpy as np
import cv2
import matplotlib.pyplot as plt

err_data = np.load(cur_path + "/download_data/data_all.npy")
pe_data = np.load(cur_path + "/download_data/condition_all.npy").squeeze(1)

r"""
每个块对应一个错误矩阵(shape:2304 * 16),数据集中每个pe对应一个块错误矩阵集合｛mat1, mat2, ... ,matn};
将集合内的错误矩阵相加，统计出该pe值下，block中每个位置的总错误次数，之后将统计情况转化为灰度图，错误次数越多，
在灰度图中颜色越趋近于黑色。
"""
def count_frequency():
    pe_set = set(list(pe_data))
    total_dict = {}
    count_dict = {}
    for pe in pe_set:
        total_dict[int(pe)] = np.zeros((2304, 16), dtype=np.int64)
        count_dict[int(pe)] = 0

    for i in range(err_data.shape[0]):
        total_dict[int(pe_data[i])] += err_data[i].astype(dtype=np.int64)
        count_dict[int(pe_data[i])] += 1

    for pe in total_dict.keys():
        total_dict[pe] = total_dict[pe].astype(np.float64)
        total_dict[pe] = (255 - (total_dict[pe] / total_dict[pe].max()) * 255).astype(np.uint8)
        cv2.imwrite(cur_path + "/count_img/pe_%s_num_%s.bmp" % (pe, count_dict[pe]), total_dict[pe])


def count_total_err_num():
    pe_set = set(list(pe_data.astype(np.int)))
    err_num_dict = {}

    for pe in pe_set:
        err_num_dict[int(pe)] = []

    for i in range(err_data.shape[0]):
        err_num_dict[int(pe_data[i])].append(int(err_data[i].sum()))

    pe_set = list(pe_set)
    pe_set.sort()
    res_set = []
    for pe in pe_set:
        res_set.append(int(sum(err_num_dict[pe]) / len(err_num_dict)))

    plt.plot(pe_set, res_set)
    plt.show()


if __name__ == "__main__":
    count_total_err_num()
