import os
import sys
cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)
sys.path.append(cur_path)

import pymysql
import numpy as np
from utils.utils import mkdir

data_root_path = "e:/nandflash_data/"
import_datebase = 0
import_local = 1
sql = "INSERT INTO tread(testID, pe, rt, rd, chip, ce, die, block, page, pagetype, tread, err, f0, f1, f2, " \
      "f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, recordtime) VALUES (%s, %s, %s, %s, %s, %s, %s, " \
      "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()) "

rt = 0
rd = 0
total_block = 0

block_err_data_list = []  # 存储整个块所有错误信息
page_err_data_list = []  # 存储块每个page的错误总数
pe_data_list = []  # 存储pe值


# 对log文件中的一个page行进行处理，返回提取到的信息
def handle_line(line):
    temp = line[1:-2].split("][")
    for i in range(len(temp)):
        temp[i] = temp[i].split(" ")
        for _ in range(temp[i].count("")):
            temp[i].remove("")

    rtn_dict = {
        "pe": temp[0][1],
        "ce": temp[0][2][1:],
        "lun": temp[0][3][1:],
        "block": temp[0][4][1:],
        "page": temp[0][5][1:],
        "page_type": temp[0][6],
        "tread": temp[1][0],
        "page_err": temp[2][0],
        "f0-15": temp[3]
    }
    return rtn_dict


# 根据handle_line提取到的多条信息组成的block，对一个block的信息进行处理，选择导入数据库或者保存到本地
def handle_block(info_list, connect, action):
    if action == import_datebase:
        with connect.cursor() as cursor:
            cursor.executemany(sql, info_list)
        connect.commit()
    else:
        block_err_data = []
        page_err_data = []

        for info in info_list:
            block_err_data.append(info[-16:])
            page_err_data.append(info[11])

        block_err_data_list.append(block_err_data)
        page_err_data_list.append(page_err_data)
        pe_data_list.append([info_list[0][1]])


# 对一个log文件进行处理
def handle_file(file_path, chip, connect, testID, action, pe_interval):
    with open(file_path) as f:
        text_content = f.readlines()
        print("%s log文件加载完毕" % file_path)

        line_num = 0
        while line_num < len(text_content):
            if text_content[line_num] == "start\n" and len(text_content[line_num + 1]) == 170:
                line_num += 2

                # 对给定的pe间隔进行导入
                pe = int(handle_line(text_content[line_num])["pe"])
                if pe % pe_interval == 0 or pe == 1:
                    block_info_list = []

                    while text_content[line_num] != "end\n":
                        info_dict = handle_line(text_content[line_num])
                        block_info_list.append(
                            (
                                int(testID),  # 0
                                int(info_dict["pe"]),  # 1
                                rt,  # 2
                                rd,  # 3
                                chip,  # 4
                                int(info_dict["ce"]),  # 5
                                int(info_dict["lun"]),  # 6
                                int(info_dict["block"]),  # 7
                                int(info_dict["page"]),  # 8
                                info_dict["page_type"],  # 9
                                int(info_dict["tread"]),  # 10
                                int(info_dict["page_err"]),  # 11
                                int(info_dict["f0-15"][0]),
                                int(info_dict["f0-15"][1]),
                                int(info_dict["f0-15"][2]),
                                int(info_dict["f0-15"][3]),
                                int(info_dict["f0-15"][4]),
                                int(info_dict["f0-15"][5]),
                                int(info_dict["f0-15"][6]),
                                int(info_dict["f0-15"][7]),
                                int(info_dict["f0-15"][8]),
                                int(info_dict["f0-15"][9]),
                                int(info_dict["f0-15"][10]),
                                int(info_dict["f0-15"][11]),
                                int(info_dict["f0-15"][12]),
                                int(info_dict["f0-15"][13]),
                                int(info_dict["f0-15"][14]),
                                int(info_dict["f0-15"][15])
                            )
                        )

                        line_num += 1

                    handle_block(block_info_list, connect, action)
                    global total_block
                    total_block += 1
                    print("%s blocks are imported" % total_block)
                else:
                    while text_content[line_num] != "end\n":
                        line_num += 1
            else:
                line_num += 1


# 导入配置文件
def import_config(connect, testID):
    description = "cycling, pe[0,20000,100]"

    sql = """insert into testgroup(groupID, testID, chip, ce, die, block, description)
    values(%s, %s, %s, %s, %s, %s, %s);
    """
    groupID = 1

    file_list = os.listdir(data_root_path)
    for file in file_list:
        if os.path.isdir(data_root_path + file):
            cur_path = data_root_path + file + "/"

            with open(cur_path + "000.log") as f:
                config_dict = {}
                for i, line in enumerate(f):
                    if line == "start\n":
                        break
                    if i == 5:
                        config_dict["block"] = line[line.find(":") + 1:-1]

                with connect.cursor() as cursor:
                    cursor.execute(sql, (groupID, testID, "0-15", "0,1,2,3", "0", config_dict["block"], description))
                connect.commit()

            groupID += 1

    print("config import done!")


# 导入数据文件
def import_data(connect, testID, action=import_datebase, pe_interval=1000):
    chip_list = list(range(16))

    if action == import_local:
        mkdir(data_root_path + "data_npy")

    file_list = os.listdir(data_root_path)
    for file in file_list:
        if os.path.isdir(data_root_path + file):
            data_cur_path = data_root_path + file + "/"
            for chip in chip_list:
                '''插入数据'''
                file_name = str(chip).zfill(3) + ".log"
                handle_file(data_cur_path + file_name, chip, connect, testID, action, pe_interval)
                with open("./import.out", "a") as f:
                    f.write("date: %s, chip: %s import success\n" % (file, chip))
                    f.write("%s blocks have been imported now\n" % total_block)

            with open("./import.out", "a") as f:
                f.write("date: %s all chips import success\n" % file)

            if action == import_local:
                global block_err_data_list, page_err_data_list, pe_data_list
                np.save(data_root_path + "data_npy/" + "block_err_data_%s.npy" % file,
                        np.array(block_err_data_list, dtype=np.int32))

                np.save(data_root_path + "data_npy/" + "page_err_data_%s.npy" % file,
                        np.array(page_err_data_list, dtype=np.int32))

                np.save(data_root_path + "data_npy/" + "pe_data_%s.npy" % file,
                        np.array(pe_data_list, dtype=np.int32))

                with open(data_root_path + "data_npy/import.log", "a") as f:
                    f.write("%s\n" % file)



def run():
    connect = pymysql.connect(host='127.0.0.1', port=3306,
                              user='root', passwd='1998msm322', db='nandflash_gan', charset='utf8mb4')
    testID = 1
    import_data(connect, testID, action=import_local, pe_interval=1000)
    connect.close()


if __name__ == "__main__":
    run()
