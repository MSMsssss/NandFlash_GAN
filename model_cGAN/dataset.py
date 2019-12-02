import torch.utils.data
from data.connect_database import Connect, SqlConfig
from model_cGAN.config import Config
import numpy as np
import torch

config = Config()


# 对数据进行归一化和正则化
def block_normalized(err_data, condition, mean=0.5, std=0.5):
    err_data = err_data / err_data.max()
    err_data = (err_data - mean) / std

    condition = condition / config.max_pe
    condition = (condition - mean) / std

    return err_data, condition


# 自定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, err_data_path="", condition_data_path="", normalize=False):
        if err_data_path == "":
            # 建立数据库连接
            self.connect = Connect(SqlConfig.train_set_database)
            # 将从数据库读出的数据保存至err_data和condition_data
            self.err_data = []
            self.condition_data = []
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
                                    data = self.connect.get_block_page_data(item["testID"], pe, chip, ce, die, block)
                                    if data is not None:
                                        self.err_data.append(data)
                                        self.condition_data.append(np.array([pe], dtype=np.float32))
                print("数据集：", item, "加载完成")

            self.err_data = np.array(self.err_data)
            self.condition_data = np.array(self.condition_data)
        else:
            self.err_data = np.load(err_data_path)
            self.condition_data = np.load(condition_data_path)

        if normalize is True:
            for i in range(self.err_data.shape[0]):
                self.err_data[i], self.condition_data[i] = block_normalized(self.err_data[i], self.condition_data[i])

    def __len__(self):
        return self.err_data.shape[0]

    def __getitem__(self, index):
        return self.err_data[index], self.condition_data[index]


# 自定义数据集
class TotalErrDataset(torch.utils.data.Dataset):
    def __init__(self, err_data_path="", condition_data_path="", normalize=False):
        # 读取的pe集合
        self.pe_set = [1] + list(range(500, 17000, 500))
        if err_data_path == "":
            # 建立数据库连接
            self.connect = Connect(SqlConfig.train_set_database)
            # 将从数据库读出的数据保存至err_data和condition_data
            self.err_data = []
            self.condition_data = []
            # 从nandflash.testgroup中得到的数据集配置信息
            self.config = self.connect.get_data_config()
            # 选择要读取的group:range[0]~range[1]范围
            self.range = (0, len(self.config))
            # self.range = (0, 1)

            print("全部数据集信息：")
            for x in self.config[self.range[0]:self.range[1]]:
                print(x)

            for item in self.config[self.range[0]:self.range[1]]:
                for chip in item["chip"]:
                    for ce in item["ce"]:
                        for die in item["die"]:
                            for block in item["block"]:
                                for pe in self.pe_set:
                                    data = self.connect.get_block_page_data(item["testID"], pe, chip, ce, die, block)
                                    if data is not None:
                                        self.err_data.append([data.sum()])
                                        self.condition_data.append(np.array([pe], dtype=np.float32))
                print("数据集：", item, "加载完成")

            self.err_data = np.array(self.err_data)
            self.condition_data = np.array(self.condition_data)
        else:
            temp_data = np.load(err_data_path)
            self.err_data = np.zeros((temp_data.shape[0], 1), dtype=np.float32)
            self.condition_data = np.load(condition_data_path)
            for i in range(self.err_data.shape[0]):
                self.err_data[i][0] = temp_data[i].sum()

        if normalize is True:
            for i in range(self.err_data.shape[0]):
                self.err_data[i] = (self.err_data[i] / config.max_total_err - 0.5) / 0.5
        print(self.err_data.shape)

        self.data_dict = {}
        for pe in self.pe_set:
            self.data_dict[pe] = []
        for i in range(self.err_data.shape[0]):
            self.data_dict[self.condition_data[i][0]].append(self.err_data[i][0])

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        pe = self.pe_set[index % len(self.pe_set)]
        index_set = np.random.randint(low=0, high=len(self.data_dict[pe]), size=(config.g_output_dim, ))
        rtn_set = []
        for i in index_set:
            rtn_set.append(self.data_dict[pe][i])
        rtn_set = np.array(rtn_set, dtype=np.float32)

        return rtn_set, np.array([pe], dtype=np.float32)
