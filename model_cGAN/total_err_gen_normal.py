"""
使用正太分布生成错误总数
"""
import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)
sys.path.append(cur_path)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from model_cGAN.config import Config
from model_cGAN.dataset import TotalErrDataset
from data.connect_database import Connect, SqlConfig

config = Config()
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="训练模型")
parser.add_argument("--eval", action="store_true", help="运行模型")
# train与eval共用参数

# train参数
parser.add_argument("--save_filename", default="total_err_gen_para.npy", help="保存的参数文件名")
# eval参数
parser.add_argument("--load_filename", default="total_err_gen_para.npy", help="加载的参数文件名")
parser.add_argument("--min_err_num", type=int, default=15000, help="设置生成假数据的下限")
parser.add_argument("--max_err_num", type=int, default=320000, help="设置生成假数据的上限")

parser.add_argument("--gen_start_pe", type=int, default=0, help="生成假数据的开始pe")
parser.add_argument("--gen_end_pe", type=int, default=17000, help="生成假数据的结束pe")
parser.add_argument("--gen_interval_pe", type=int, default=500, help="生成假数据的间隔pe")
parser.add_argument("--generator_data_num", type=int, default=3000,
                    help="每个pe生成generator_data_num个数据")
opt = parser.parse_args()
print(opt)


def train():
    err_data = np.load(cur_path + "/download_data/data_all.npy")
    pe_data = np.load(cur_path + "/download_data/condition_all.npy").squeeze(1)

    total_err_data = {}
    for i in range(err_data.shape[0]):
        if int(pe_data[i]) not in total_err_data:
            total_err_data[int(pe_data[i])] = []
        else:
            total_err_data[int(pe_data[i])].append(err_data[i].sum())

    std_set = []
    mean_set = []

    for pe in config.pe_set:
        total_err_data[pe] = np.array(total_err_data[pe])
        mean_set.append(total_err_data[pe].mean())
        std_set.append(total_err_data[pe].std())

    pe_set = np.array(config.pe_set).reshape((1, len(config.pe_set)))
    mean_set = np.array(mean_set).reshape((1, len(mean_set)))
    std_set = np.array(std_set).reshape((1, len(std_set)))

    para = np.concatenate((pe_set, mean_set, std_set), axis=0)
    np.save(config.model_saved_path + opt.save_filename, para)


def model_eval():
    gen_data_path = cur_path + "/gen_data/total_err_gen"
    para = np.load(config.model_saved_path + opt.load_filename)
    pe_set = para[0]
    mean_set = para[1]
    std_set = para[2]

    # 对样本的均值和标准差进行三次插值
    func_mean = interp1d(pe_set, mean_set, kind="cubic", fill_value=(mean_set[0], mean_set[-1]), bounds_error=False)
    func_std = interp1d(pe_set, std_set, kind="cubic", fill_value=(std_set[0], std_set[-1]), bounds_error=False)

    sample_pe = np.array(list(range(opt.gen_start_pe, opt.gen_end_pe + opt.gen_interval_pe, opt.gen_interval_pe)))
    sample_mean = func_mean(sample_pe)
    sample_std = func_std(sample_pe)

    for i in range(sample_pe.shape[0]):
        pe = sample_pe[i]
        mean = sample_mean[i]
        std = sample_std[i] * 0.8
        num = opt.generator_data_num

        sample_err_num = (np.random.randn(num) * std + mean).astype(dtype=np.int32)
        np.save(gen_data_path + "/total_err_num_%s" % pe, sample_err_num)


def run():
    if opt.train and opt.eval:
        print("error")
    elif opt.train:
        train()
    elif opt.eval:
        model_eval()


if __name__ == "__main__":
    run()
