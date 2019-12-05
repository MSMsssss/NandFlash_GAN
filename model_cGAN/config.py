import os
import sys


class Config:
    def __init__(self):
        self.width = 16  # Block数据的宽度，对应一个Page的f0 ~ f15
        self.height = 2304  # Block数据的高度， 对应一个Block的2304个Page
        self.condition_dim = 1  # 输入条件的维度
        self.betas = (0.5, 0.999)  # 用于计算梯度以及梯度平方的运行平均值的系数
        self.max_pe = 17000
        self.pe_interval = 1000
        self.max_total_err = 320000
        self.max_page_err = 200
        self.pe_set = [1] + list(range(self.pe_interval, self.max_pe + self.pe_interval, self.pe_interval))
        self.model_saved_path = os.path.dirname(os.path.abspath(__file__)) + "/save_model/"
        self.g_output_dim = 2000  # total_err_num 生成器输出维度
