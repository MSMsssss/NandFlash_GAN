import torch.utils.data
from data.connect_database import Connect
import numpy as np


class Config(object):
    def __init__(self):
        self.testID = 4  # 测试编号
        self.chip = list(range(16))  # 芯片编号
        self.ce = list(range(4))  # ce编号
        self.die = [0]  # die编号
        self.block = [2, 3]  # 块编号
        self.pe_set = [1] + list(range(100, 15100, 100))


# 自定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.connect = Connect("nandflash")
        self.config = Config()
        self.data_set = []

        for chip in self.config.chip:
            for ce in self.config.ce:
                for die in self.config.die:
                    for block in self.config.block:
                        for pe in self.config.pe_set:
                            self.data_set.append(
                                (self.connect.get_block_data(
                                    self.config.testID, pe, chip, ce, die, block), np.array([pe], dtype=np.float32)))

                            print("testID: %s, pe: %s, chip: %s, ce: %s, die: %s, block: %s 加载完成" %
                                  (self.config.testID, pe, chip, ce, die, block))

    def __len__(self):
        return len(self.config.chip) * len(self.config.ce) * \
               len(self.config.die) * len(self.config.block) * len(self.config.pe_set)

    def __getitem__(self, index):
        return self.data_set[index]
