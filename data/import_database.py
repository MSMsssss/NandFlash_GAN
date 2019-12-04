# coding=utf-8
import pymysql
import datetime
import sys
import os
import numpy

data_root_path = "e:/nandflash_data/"
import_datebase = 0
import_local = 1
sql = "INSERT INTO tread(testID, pe, rt, rd, chip, ce, die, block, page, pagetype, tread, err, f0, f1, f2, " \
      "f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, recordtime) VALUES (%s, %s, %s, %s, %s, %s, %s, " \
      "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()) "

rt = 0
rd = 0
total_block = 0


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
def handle_block(info_list, connect, action=import_datebase):
    if action == import_datebase:
        with connect.cursor() as cursor:
            cursor.executemany(sql, info_list)
        connect.commit()
    else:
        print("")


# 对一个log文件进行处理
def handle_file(file_path, chip, connect, testID, action=import_datebase):
    with open(file_path) as f:
        text_content = f.readlines()
        print("%s log文件加载完毕" % file_path)
        line_num = 0
        while line_num < len(text_content):
            if text_content[line_num] == "start\n" and len(text_content[line_num + 1]) == 170:
                line_num += 2

                block_info_list = []
                while text_content[line_num] != "end\n":
                    info_dict = handle_line(text_content[line_num])
                    block_info_list.append(
                        (
                            testID,
                            info_dict["pe"],
                            rt,
                            rd,
                            chip,
                            info_dict["ce"],
                            info_dict["lun"],
                            info_dict["block"],
                            info_dict["page"],
                            info_dict["page_type"],
                            info_dict["tread"],
                            info_dict["page_err"],
                            info_dict["f0-15"][0],
                            info_dict["f0-15"][1],
                            info_dict["f0-15"][2],
                            info_dict["f0-15"][3],
                            info_dict["f0-15"][4],
                            info_dict["f0-15"][5],
                            info_dict["f0-15"][6],
                            info_dict["f0-15"][7],
                            info_dict["f0-15"][8],
                            info_dict["f0-15"][9],
                            info_dict["f0-15"][10],
                            info_dict["f0-15"][11],
                            info_dict["f0-15"][12],
                            info_dict["f0-15"][13],
                            info_dict["f0-15"][14],
                            info_dict["f0-15"][15]
                        )
                    )

                    line_num += 1

                handle_block(block_info_list, connect)
                global total_block
                total_block += 1
                print("%s blocks are imported" % total_block)
            else:
                line_num += 1


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


def import_data(connect, testID, action=import_datebase):
    chips = list(range(16))

    file_list = os.listdir(data_root_path)
    for file in file_list:
        if os.path.isdir(data_root_path + file):
            cur_path = data_root_path + file + "/"
            for item in chips:
                '''插入数据'''
                file_name = str(item).zfill(3) + ".log"
                handle_file(cur_path + file_name, item, connect, testID)
                print("date: %s, chip %s import success" % (file, item))

            print("date: %s log import success" % file)


def run():
    connect = pymysql.connect(host='127.0.0.1', port=3306,
                              user='root', passwd='1998msm322', db='nandflash_gan', charset='utf8mb4')
    testID = 1
    import_data(connect, testID, import_datebase)
    connect.close()


if __name__ == "__main__":
    run()
