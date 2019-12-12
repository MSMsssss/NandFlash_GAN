import os
import sys
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def norm_ip(img, min, max):
    img = img.clip(min, max)
    img = (img - min) / (max - min)
    return img


def norm_range(t, range=None):
    if range is not None:
        return norm_ip(t, float(range[0]), float(range[1]))
    else:
        return norm_ip(t, float(t.min()), float(t.max()))


# 随机生成总和为sum的num个随机数
def random_int_with_sum(sum, num):
    # 随机生成num-1个随机数作为分割点
    split_point_set = [0] + list(np.random.randint(low=0, high=sum, size=(num-1, ))) + [sum]
    split_point_set.sort()

    result = []
    for i in range(num):
        result.append(split_point_set[i+1] - split_point_set[i])

    return np.array(result, dtype=np.int32)


if __name__ == "__main__":
    result = random_int_with_sum(88, 16)
    print(result, result.shape[0], result.sum())
