import torch.utils.data
from data.connect_database import Connect, SqlConfig
import numpy as np
import torch


# class Config(object):
#     def __init__(self):
#         # 在拥有GPU的服务器上运行时读取较多的数据
#         if not torch.cuda.is_available():
#             self.testID = 4  # 测试编号
#             self.chip = list(range(1))  # 芯片编号
#             self.ce = list(range(1))  # ce编号
#             self.die = [0]  # die编号
#             self.block = [2, 3]  # 块编号
#             self.pe_set = [1] + list(range(13000, 15100, 1000))
#         else:
#             self.testID = [4, 5]  # 测试编号
#             self.chip = list(range(16))  # 芯片编号
#             self.ce = list(range(4))  # ce编号
#             self.die = [0]  # die编号
#             self.block = {4: [2, 3], 5: [4, 5, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}  # 块编号
#             self.pe_set = [1] + list(range(1000, 17000, 1000))


# 自定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.connect = Connect(SqlConfig.train_set_database)
        self.data_set = []
        self.config = self.connect.get_data_config()
        self.pe_set = [1] + list(range(1000, 17000, 1000))

        print("全部数据集信息：")
        for x in self.config:
            print(x)

        for item in self.config:
            for chip in item["chip"]:
                for ce in item["ce"]:
                    for die in item["die"]:
                        for block in item["block"]:
                            for pe in self.pe_set:
                                data = self.connect.get_block_data(item["testID"], pe, chip, ce, die, block)
                                if data is not None:
                                    self.data_set.append((data, np.array([pe], dtype=np.float32)))

            print("数据集：", item, "加载完成")

    def __len__(self):
        return len(self.config.chip) * len(self.config.ce) * \
               len(self.config.die) * len(self.config.block) * len(self.config.pe_set)

    def __getitem__(self, index):
        return self.data_set[index]
