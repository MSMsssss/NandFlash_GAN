import pymysql
import numpy as np
import torch


class SqlConfig(object):
    train_set_database = "nandflash_gan"
    generator_database = "nandflash_gan"

    def __init__(self, database):
        self.host = '127.0.0.1'  # 主机地址
        self.user = 'root'  # 用户名
        self.password = '1998msm322'  # 密码
        self.db = database  # 数据库名称
        self.charset = 'utf8mb4'  # 字符集
        self.cursorclass = pymysql.cursors.DictCursor  # 游标类型
        self.page_num = 2304  # 每个块含有的page数目
        self.f_num = 16
        self.page_type = {"SLC": 0, "MLC_LP": 1, "MLC_UP": 2, "TLC_LP": 3, "TLC_UP": 4, "TLC_EX": 5}


def string_handle(string):
    if string.find("-") == -1:
        return [int(s) for s in string.replace(" ", "").split(",")]
    else:
        index = [int(s) for s in string.replace(" ", "").split("-")]
        return list(range(index[0], index[1] + 1))


class Connect:
    def __init__(self, database):
        self.config = SqlConfig(database)
        self.connection = pymysql.connect(host=self.config.host,
                                          user=self.config.user,
                                          password=self.config.password,
                                          db=self.config.db,
                                          charset=self.config.charset,
                                          cursorclass=self.config.cursorclass
                                          )

        self.sql_get_block = """select f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15 
        from tread where testID = %s and pe = %s and chip = %s and ce = %s and die = %s and block = %s
        """

        self.sql_get_block_page_info = """select pagetype from tread where 
        testID = %s and pe = %s and chip = %s and ce = %s and die = %s and block = %s
        """

        self.sql_get_block_page_err_info = """select err from tread where 
        testID = %s and pe = %s and chip = %s and ce = %s and die = %s and block = %s
        """

        self.sql_insert_page = """insert into generator_data(
        block_id, page_id, err, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15)
        values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        self.sql_sum_block_err = """update blocks set total_err = 
        (select sum(err) from generator_data where block_id = %s)
        where block_id = %s
        """

    def __del__(self):
        self.connection.close()

    # 从nandflash.testgroup中获得训练数据的配置
    def get_data_config(self):
        # 执行此函数必须使用训练集数据库
        if self.config.db != SqlConfig.train_set_database:
            raise RuntimeError("used error database!")

        with self.connection.cursor() as cursor:
            cursor.execute("select * from testgroup")
            group_info = cursor.fetchall()

        config = []
        for item in group_info:
            info_dict = {"testID": item["testID"], "chip": string_handle(item["chip"]), "ce": string_handle(item["ce"]),
                         "die": string_handle(item["die"]), "block": string_handle(item["block"])}

            config.append(info_dict)

        return config

    # 获得指定block的2304个page每个page的错误总数
    def get_block_page_data(self, testID, pe, chip, ce, die, block):
        # 执行此函数必须使用训练集数据库
        if self.config.db != SqlConfig.train_set_database:
            raise RuntimeError("used error database!")

        with self.connection.cursor() as cursor:
            cursor.execute(self.sql_get_block_page_err_info, (testID, pe, chip, ce, die, block))
            data = cursor.fetchall()
            if len(data) != self.config.page_num:
                return None

            result = np.zeros((self.config.page_num, ), dtype=np.float32)

            for page_index in range(self.config.page_num):
                    result[page_index] = data[page_index]["err"]

            return result

    # 从nandflash.tread数据库获得块错误信息
    def get_block_data(self, testID, pe, chip, ce, die, block):
        # 执行此函数必须使用训练集数据库
        if self.config.db != SqlConfig.train_set_database:
            raise RuntimeError("used error database!")

        with self.connection.cursor() as cursor:
            cursor.execute(self.sql_get_block, (testID, pe, chip, ce, die, block))
            data = cursor.fetchall()
            if len(data) != self.config.page_num:
                return None

            result = np.zeros((self.config.page_num, self.config.f_num), dtype=np.float32)

            for page_index in range(self.config.page_num):
                for f_index in range(self.config.f_num):
                    result[page_index][f_index] = data[page_index]["f" + str(f_index)]

            return result

    # 向NandFlash_GAN数据库插入1个block的fake数据
    def insert_block_data(self, fake_data, pe):
        # 执行此函数必须使用保存生成数据的数据库
        if self.config.db != SqlConfig.generator_database:
            raise RuntimeError("used error database!")

        fake_data = fake_data[0].type(torch.int32).to("cpu").apply_(lambda a: a if a > 0 else 0)
        with self.connection.cursor() as cursor:
            cursor.execute("insert into blocks(pe, total_err, submit_date) values (%s, %s, now())", (pe, 0))
        self.connection.commit()

        with self.connection.cursor() as cursor:
            cursor.execute("select @new_block_id");
            new_block_id = cursor.fetchone()['@new_block_id']

        for page_id in range(self.config.page_num):
            with self.connection.cursor() as cursor:
                cursor.execute(self.sql_insert_page,
                               (new_block_id, page_id, fake_data[page_id].sum().item(),
                                fake_data[page_id][0].item(),
                                fake_data[page_id][1].item(),
                                fake_data[page_id][2].item(),
                                fake_data[page_id][3].item(),
                                fake_data[page_id][4].item(),
                                fake_data[page_id][5].item(),
                                fake_data[page_id][6].item(),
                                fake_data[page_id][7].item(),
                                fake_data[page_id][8].item(),
                                fake_data[page_id][9].item(),
                                fake_data[page_id][10].item(),
                                fake_data[page_id][11].item(),
                                fake_data[page_id][12].item(),
                                fake_data[page_id][13].item(),
                                fake_data[page_id][14].item(),
                                fake_data[page_id][15].item()
                                ))

            self.connection.commit()

        with self.connection.cursor() as cursor:
            cursor.execute(self.sql_sum_block_err, (new_block_id, new_block_id))
        self.connection.commit()

    # 获取块的pagetype信息
    def get_block_page_info(self, testID, pe, chip, ce, die, block):
        # 执行此函数必须使用训练集数据库
        if self.config.db != SqlConfig.train_set_database:
            raise RuntimeError("used error database!")

        with self.connection.cursor() as cursor:
            cursor.execute(self.sql_get_block_page_info, (testID, pe, chip, ce, die, block))
            data = cursor.fetchall()
            if len(data) != self.config.page_num:
                return None

            result = np.zeros((self.config.page_num,), dtype=np.int32)

            for page_index in range(self.config.page_num):
                    result[page_index] = self.config.page_type[data[page_index]["pagetype"]]

            return result


if __name__ == "__main__":
    connect = Connect(SqlConfig.train_set_database)
    for x in connect.get_data_config():
        print(x)
