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

err_data = np.load(cur_path + "/download_data/data_all.npy")
pe_data = np.load(cur_path + "/download_data/condition_all.npy").squeeze(1)
# err_data = np.load(cur_path + "/gen_data/gen_data_60.npy")
# pe_data = np.load(cur_path + "/gen_data/gen_condition_60.npy").squeeze(1)


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
            img = 255 - err_data[i] * 255
            id += 1
            cv2.imwrite(cur_path + "/count_img/%s/pe_%s_id_%s.bmp" % (dset, pe, id), img)


# 统计生成的fake数据的分布，并转换器灰度图
def count_gen_data():
    pe_set = set(list(pe_data))
    total_dict = {}
    count_dict = {}
    for pe in pe_set:
        total_dict[int(pe)] = np.zeros((2304, 16), dtype=np.float64)
        count_dict[int(pe)] = 0

    for i in range(err_data.shape[0]):
        total_dict[int(pe_data[i])] += err_data[i].astype(dtype=np.float64)
        count_dict[int(pe_data[i])] += 1

    for pe in total_dict.keys():
        total_dict[pe] = norm_range(total_dict[pe])
        total_dict[pe] = (255 - total_dict[pe] * 255).astype(np.uint8)
        cv2.imwrite(cur_path + "/count_img/fake/pe_%s_num_%s.bmp" % (pe, count_dict[pe]), total_dict[pe])


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
        total_dict[int(pe)] = np.zeros((2304, 16), dtype=np.int64)
        count_dict[int(pe)] = 0

    for i in range(err_data.shape[0]):
        total_dict[int(pe_data[i])] += err_data[i].astype(dtype=np.int64)
        count_dict[int(pe_data[i])] += 1

    for pe in total_dict.keys():
        total_dict[pe] = total_dict[pe].astype(np.float64) / count_dict[pe]
        total_dict[pe] = norm_range(total_dict[pe], (0, 15))
        total_dict[pe] = (255 - total_dict[pe] * 255).astype(np.uint8)
        cv2.imwrite(cur_path + "/count_img/real/pe_%s_num_%s.bmp" % (pe, count_dict[pe]), total_dict[pe])


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


def real_data_para():
    max_err = []
    for i in range(err_data.shape[0]):
        max_err.append(err_data[i].max())

    max_err = np.array(max_err)
    print("min: %s, max: %s, mean: %s, std: %s" % (max_err.min(), max_err.max(), max_err.mean(), max_err.std()))
    num = 0
    for i in range(max_err.shape[0]):
        if max_err[i] > 20:
            num += 1
            print(max_err[i])
    print(num, num / max_err.shape[0])

    t = torch.from_numpy(err_data).apply_(lambda x: 1 if x > 15 else 0)
    total = t.sum()

    print(total, err_data.shape[0] * err_data.shape[1] * err_data.shape[2],
          total / (err_data.shape[0] * err_data.shape[1] * err_data.shape[2]))


# 将块的2304个page的page类型映射为图片
def count_page_info():
    connect = Connect(SqlConfig.train_set_database)
    rtn = connect.get_block_page_info(3, 1, 0, 0, 0, 0)
    # 黄色， 黑色， 红色， 白色， 绿色， 蓝色
    color_dict = {0: (0, 255, 255), 1: (0, 0, 0), 2: (0, 0, 255), 3: (255, 255, 255), 4: (0, 255, 0), 5: (255, 0, 0)}
    img = np.zeros((2304, 64, 3), dtype=np.uint8)
    for i in range(2304):
        for j in range(64):
            img[i][j] = color_dict[rtn[i]]

    cv2.imwrite(cur_path + "/count_img/pagetype.jpg", img)


# 计算真实数据分布在每一行上的离散程度
def count_row_std(pe):
    total = np.zeros((2304, 16), dtype=np.float64)
    count = 0
    for i in range(err_data.shape[0]):
        if pe_data[i] == pe:
            total += err_data[i]
            count += 1

    total = total / count
    std = np.zeros([2304,], dtype=np.float64)
    for i in range(2304):
        std[i] = total[i].std()

    print("pe: %s, mean_err: %s, row_std_min: %s, row_std_max: %s, row_std_mean: %s" %
          (pe, total.mean(), std.min(), std.max(), std.mean()))


if __name__ == "__main__":
    for pe in [1] + list(range(500, 16500, 500)):
        count_row_std(pe)
