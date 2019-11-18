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
        # 建立数据库连接
        self.connect = Connect(SqlConfig.train_set_database)
        # 将从数据库读出的数据全部保存至data_set
        self.data_set = []
        # 从nandflash.testgroup中得到的数据集配置信息
        self.config = self.connect.get_data_config()
        # 选择要读取的group范围
        self.range = (0, len(self.config))
        # 读取的pe集合
        self.pe_set = [1] + list(range(1000, 17000, 1000))

        print("全部数据集信息：")
        for x in self.config[self.range[0]:self.range[1]]:
            print(x)

        for item in self.config[self.range[0]:self.range[1]]:
            for chip in item["chip"]:
                for ce in item["ce"]:
                    for die in item["die"]:
                        for block in item["block"]:
                            for pe in self.pe_set:
                                data = self.connect.get_block_data(item["testID"], pe, chip, ce, die, block)
                                if data is not None:
                                    self.data_set.append((data, np.array([pe], dtype=np.float32)))
                                    # print("chip:", chip, "ce:", ce, "die", die, "block:", block, "pe:", pe, "加载完成")

            print("数据集：", item, "加载完成")

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]
