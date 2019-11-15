import os
import sys


class Config:
    def __init__(self):
        self.width = 16  # Block数据的宽度，对应一个Page的f0 ~ f15
        self.height = 2304  # Block数据的高度， 对应一个Block的2304个Page
        self.latent_dim = 100  # 输入噪声的维度
        self.condition_dim = 1  # 输入条件的维度
        self.betas = (0.9, 0.999)  # 用于计算梯度以及梯度平方的运行平均值的系数
        self.epochs = 100  # 训练轮数
        self.batch_size = 32
        self.lr = 0.02  # 学习速率
        self.pe_set = [1] + list(range(100, 15100, 100))
        self.save_model_epoch = 50  # 设置每隔多少轮保存一次模型
        self.model_saved_path = os.path.dirname(os.path.abspath(__file__)) + "/save_model/"
