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


class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.page_num = 2304
        self.f_num = 16
        self.data = []
        self.pe_set = list(range(0, 15000, 500))
        for pe in self.pe_set:
            for _ in range(1000):
                self.data.append((np.ones((self.page_num, self.f_num), dtype=np.float32) * np.exp(pe /1500),
                                  np.array([pe], dtype=np.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def block_normalized(block):
    block = block / block.max()

    return block

# 自定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, err_data_path="", condition_data_path=""):
        self.max_pe = 17000
        if err_data_path == "":
            self.load_from_local = False
            # 建立数据库连接
            self.connect = Connect(SqlConfig.train_set_database)
            # 将从数据库读出的数据全部保存至data_set
            self.data_set = []
            # 从nandflash.testgroup中得到的数据集配置信息
            self.config = self.connect.get_data_config()
            # 选择要读取的group:range[0]~range[1]范围
            self.range = (0, len(self.config))
            # self.range = (0, 1)
            # 读取的pe集合
            self.pe_set = [1] + list(range(500, 17000, 500))

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

            for i in range(len(self.data_set)):
                self.data_set[i][0] = block_normalized(self.data_set[i][0])
                self.data_set[i][1] = self.data_set[i][1] / self.max_pe
        else:
            self.load_from_local = True
            self.err_data = np.load(err_data_path)
            self.condition_data = np.load(condition_data_path)

            for i in range(self.err_data.shape[0]):
                self.err_data[i] = block_normalized(self.err_data[i])
                self.condition_data[i] = self.condition_data[i] / self.max_pe

    def __len__(self):
        if self.load_from_local:
            return self.err_data.shape[0]
        else:
            return len(self.data_set)

    def __getitem__(self, index):
        if self.load_from_local:
            return self.err_data[index], self.condition_data[index]
        else:
            return self.data_set[index]
