import os
import sys


class Config:
    def __init__(self):
        self.width = 16  # Block数据的宽度，对应一个Page的f0 ~ f15
        self.height = 2304  # Block数据的高度， 对应一个Block的2304个Page
        self.condition_dim = 1  # 输入条件的维度
        self.betas = (0.5, 0.999)  # 用于计算梯度以及梯度平方的运行平均值的系数
        self.max_pe = 17000
        self.pe_set = [1] + list(range(500, self.max_pe, 500))
        self.model_saved_path = os.path.dirname(os.path.abspath(__file__)) + "/save_model/"
