import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cur_path = root_path + "/data"

import numpy as np
import cv2

err_data = np.load(cur_path + "/download_data/data_all.npy")
pe_data = np.load(cur_path + "/download_data/condition_all.npy").squeeze(1)


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
