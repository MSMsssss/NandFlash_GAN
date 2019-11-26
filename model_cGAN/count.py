import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(cur_path)
sys.path.append(root_path)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from data.connect_database import Connect, SqlConfig

# err_data = np.load(cur_path + "/download_data/data_all.npy")
# pe_data = np.load(cur_path + "/download_data/condition_all.npy").squeeze(1)
err_data = np.load(cur_path + "/gen_data/gen_data_200.npy")
pe_data = np.load(cur_path + "/gen_data/gen_condition_200.npy").squeeze(1)


def norm_ip(img, min, max):
    img = img.clip(min, max)
    img = (img - min) / (max - min + 1e-5)
    return img


def norm_range(t, range=None):
    if range is not None:
        return norm_ip(t, float(range[0]), float(range[1]))
    else:
        return norm_ip(t, float(t.min()), float(t.max()))


# 显示单个block的错误分布，并转换为灰度图
def show_gen_data_pe(pe, dset='real'):
    id = 0
    for i in range(err_data.shape[0]):
        if int(pe_data[i]) == pe:
            if dset != 'real':
                err_data[i] = err_data[i] - err_data[i].min()
            err_data[i] = err_data[i] / err_data[i].max()
            err_data[i] = 255 - err_data[i] * 255
            img = np.zeros((2304, 32), dtype=np.uint8)
            for j in range(2304):
                img[j] = 32 * [err_data[i][j]]
            id += 1
            cv2.imwrite(cur_path + "/count_img/%s/pe_%s_id_%s.bmp" % (dset, pe, id), img)


# 统计生成的fake数据的分布，并转换器灰度图
def count_gen_data():
    pe_set = set(list(pe_data))
    total_dict = {}
    count_dict = {}
    for pe in pe_set:
        total_dict[int(pe)] = np.zeros((2304,), dtype=np.float64)
        count_dict[int(pe)] = 0

    for i in range(err_data.shape[0]):
        total_dict[int(pe_data[i])] += err_data[i].astype(dtype=np.float64)
        count_dict[int(pe_data[i])] += 1

    for pe in total_dict.keys():
        total_dict[pe] = total_dict[pe] / count_dict[pe]
        total_dict[pe] = norm_range(total_dict[pe])
        total_dict[pe] = (255 - total_dict[pe] * 255).astype(np.uint8)
        img = np.zeros((2304, 32), dtype=np.uint8)
        for i in range(2304):
            img[i] = 32 * [total_dict[pe][i]]
        cv2.imwrite(cur_path + "/count_img/fake/pe_%s_num_%s.bmp" % (pe, count_dict[pe]), img)


def count_frequency():
    r"""
    每个块对应一个错误矩阵(shape:2304 * 16),数据集中每个pe对应一个块错误矩阵集合｛mat1, mat2, ... ,matn};
    将集合内的错误矩阵相加，统计出该pe值下，block中每个位置的总错误次数，之后将统计情况转化为灰度图，错误次数越多，
    在灰度图中颜色越趋近于黑色。
    """

    pe_set = set(list(pe_data))
    total_dict = {}
    count_dict = {}
    for pe in pe_set:
        total_dict[int(pe)] = np.zeros((2304,), dtype=np.int64)
        count_dict[int(pe)] = 0

    for i in range(err_data.shape[0]):
        total_dict[int(pe_data[i])] += err_data[i].astype(dtype=np.int64)
        count_dict[int(pe_data[i])] += 1

    for pe in total_dict.keys():
        total_dict[pe] = total_dict[pe].astype(np.float64) / count_dict[pe]
        total_dict[pe] = norm_range(total_dict[pe])
        total_dict[pe] = (255 - total_dict[pe] * 255).astype(np.uint8)
        img = np.zeros((2304, 32), dtype=np.uint8)
        for i in range(2304):
            img[i] = 32 * [total_dict[pe][i]]
        cv2.imwrite(cur_path + "/count_img/real/pe_%s_num_%s.bmp" % (pe, count_dict[pe]), img)


# 统计块错误总数与pe的关系
def count_total_err_num():
    pe_set = set(list(pe_data.astype(np.int)))
    err_num_dict = {}
    count_dict = {}

    for pe in pe_set:
        err_num_dict[int(pe)] = 0
        count_dict[int(pe)] = 0

    for i in range(err_data.shape[0]):
        err_num_dict[int(pe_data[i])] += err_data[i].sum()
        count_dict[int(pe_data[i])] += 1

    pe_set = list(pe_set)
    pe_set.sort()
    res_set = []

    for pe in pe_set:
        res_set.append(err_num_dict[pe] / count_dict[pe])

    plt.plot(pe_set, res_set)
    plt.show()


if __name__ == "__main__":
    count_gen_data()
