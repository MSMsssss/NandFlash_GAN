import pymysql
import numpy as np
import torch


class SqlConfig(object):
    def __init__(self, database):
        self.host = '10.147.20.88'  # 主机地址
        self.user = 'root'  # 用户名
        self.password = 'nvmgroup'  # 密码
        self.db = database  # 数据库名称
        self.charset = 'utf8mb4'  # 字符集
        self.cursorclass = pymysql.cursors.DictCursor  # 游标类型
        self.page_num = 2304  # 每个块含有的page数目
        self.f_num = 16


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

        self.sql_insert_page = """insert into generator_data(
        block_id, page_id, err, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15)
        values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        self.sql_sum_block_err = """update blocks set total_err = 
        (select sum(err) from generator_data where block_id = %s)
        """

    def __del__(self):
        self.connection.close()

    # 从nandflash数据库获得块错误信息
    def get_block_data(self, testID, pe, chip, ce, die, block):
        with self.connection.cursor() as cursor:
            cursor.execute(self.sql_get_block, (testID, pe, chip, ce, die, block))
            data = cursor.fetchall()
            result = np.zeros((self.config.page_num, self.config.f_num), dtype=np.float32)

            for page_index in range(self.config.page_num):
                for f_index in range(self.config.f_num):
                    result[page_index][f_index] = data[page_index]["f" + str(f_index)]

            return result

    # 向NandFlash_GAN数据库插入1个block的fake数据
    def insert_block_data(self, fake_data, pe):
        fake_data = fake_data[0].type(torch.int32).apply_(lambda a: a if a > 0 else 0)
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
            cursor.execute(self.sql_sum_block_err, (new_block_id, ))
        self.connection.commit()
