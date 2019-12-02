# coding=utf-8
import pymysql
import datetime
import sys
data_path = "e:/nandflash_data/2019_8_1/"


def read_file(file_path, chip, db, testID, start_lines, exit_lines):
    i = 0
    health = 0
    modelID = 1
    rt = 0
    rd = 0
    block_err = []
    k = 5000
    pe_number = 0
    blockID_list = []

    with open(file_path) as f:
        line = f.readline()
        read_recordID = 0

        while line:
            if i % 5000 == 0:
                print(i)
            # if i==exit_lines:
            # sys.exit(0)

            if i >= start_lines:
                health = 0

                read_record = []

                line = f.readline()
                x = line[1:-2].split("][")
                # print(x)
                x1 = x[0].split(" ")
                for j in range(x1.count('')):
                    x1.remove('')

                '''
                向tread表中插入数据
                从blockstatus数据表中提取pe和health值，health值判断block有没有坏，pe作为该block下read的pe次数，
                并且计算blockbe表中block_err:取2304个16K页中err最高的前115个err的平均值
                '''
                if line == "[Group PEcnt CE LUN Block Page  Type  ][ tR ][page_err][F[0]   F[1]   F[2]   F[3]   F[4] "\
                           " F[5]   F[6]   F[7]   F[8]   F[9]   F[a]   F[b]   F[c]   F[d]   F[e]   F[f]   ]\n":
                    block_err = []
                    k = 0

                if len(x1) == 7 and x1[0] != "Group" and len(x) == 4:

                    x2 = x[3].split(" ")
                    for j in range(x2.count('')):
                        x2.remove('')

                    if "read" in x2:
                        health = 3
                        print(x2)

                    for j in x1:
                        j.strip()

                    x[1].strip()
                    x[2].strip()

                    if health == 0:
                        block_err.append(int(x[2]))

                    k += 1

                    '''
                    [Group PEcnt CE LUN Block Page  Type  ][ tR ][page_err][F[0]   F[1]   F[2]   F[3]   F[4]   F[5]   F[6]   F[7]   F[8]   F[9]   F[a]   F[b]   F[c]   F[d]   F[e]   F[f]   ]
                    [20000     1 C0 L0  B20   P0    SLC   ][  79][       0][0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      ]
                    '''
                    read_record.append(read_recordID)  # recordID        read_record[0]
                    read_record.append(testID)  # testID=1        read_record[1]
                    read_record.append(int(x1[1]))  # pe              read_record[2]
                    read_record.append(rt)  # rt              read_record[3]
                    read_record.append(rd)  # rd              read_record[4]
                    read_record.append(chip)  # chip            read_record[5]
                    read_record.append(int(x1[2][1:]))  # ce              read_record[6]
                    read_record.append(int(x1[3][1:]))  # die             read_record[7]
                    read_record.append(int(x1[4][1:]))  # block           read_record[8]
                    read_record.append(int(x1[5][1:]))  # page            read_record[9]
                    read_record.append(x1[6])  # page_type       read_record[10]
                    if health == 0:
                        read_record.append(int(x[1]))  # tread           read_record[11]
                    else:
                        read_record.append(0)
                    if health == 0:
                        read_record.append(int(x[2]))  # err             read_record[12]
                    else:
                        read_record.append(0)  # err             read_record[12]
                    read_record.append(int(x2[0]))  # f0              read_record[13]
                    read_record.append(int(x2[1]))  # f1              read_record[14]
                    read_record.append(int(x2[2]))  # f2              read_record[15]
                    read_record.append(int(x2[3]))  # f3              read_record[16]
                    read_record.append(int(x2[4]))  # f4              read_record[17]
                    read_record.append(int(x2[5]))  # f5              read_record[18]
                    read_record.append(int(x2[6]))  # f6              read_record[19]
                    read_record.append(int(x2[7]))  # f7              read_record[20]
                    read_record.append(int(x2[8]))  # f8              read_record[21]
                    read_record.append(int(x2[9]))  # f9              read_record[22]
                    read_record.append(int(x2[10]))  # f10             read_record[23]
                    read_record.append(int(x2[11]))  # f11             read_record[24]
                    read_record.append(int(x2[12]))  # f12             read_record[25]
                    read_record.append(int(x2[13]))  # f13             read_record[26]
                    read_record.append(int(x2[14]))  # f14             read_record[27]
                    read_record.append(int(x2[15]))  # f15             read_record[28]
                    recordtime = datetime.datetime.now()  # recordtime      read_record[29]
                    read_record.append(recordtime)

                    blockID = str(chip).zfill(3) + str(x1[2][1:].zfill(2)) + str(x1[3][1:].zfill(2)) + str(
                        x1[4][1:].zfill(4))

                    sql = "SELECT pe,health FROM blockstatus WHERE blockID = '%s'" % (blockID)
                    cursor = db.cursor()
                    try:
                        cursor.execute(sql)
                        results = cursor.fetchall()
                        # print("results:",results)
                    except Exception as e:
                        db.rollback()  # 发生错误时回滚
                        print(e)
                    cursor.close()

                    if results[0][1] == 0:
                        '''
                        print("tread:", read_record[0], read_record[1], read_record[2], read_record[3], read_record[4],
                              read_record[5], read_record[6], read_record[7], read_record[8], read_record[9],
                              read_record[10], read_record[11], read_record[12], read_record[13], read_record[14],
                              read_record[15], read_record[16], read_record[17], read_record[18], read_record[19],
                              read_record[20], read_record[21], read_record[22], read_record[23], read_record[24],
                              read_record[25], read_record[26], read_record[27], read_record[28], read_record[29])
                        '''
                        sql = "INSERT INTO tread(testID,pe,rt,rd,chip,ce,die,block,page,pagetype,tread,err,f0,f1,f2," \
                              "f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,recordtime) VALUES (%s,%s,%s,%s,%s,%s,%s," \
                              "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "

                        cursor = db.cursor()

                        try:
                            cursor.execute(sql, (
                                read_record[1], results[0][0], read_record[3], read_record[4], read_record[5],
                                read_record[6],
                                read_record[7], read_record[8], read_record[9], read_record[10], read_record[11],
                                read_record[12],
                                read_record[13], read_record[14], read_record[15], read_record[16], read_record[17],
                                read_record[18], read_record[19], read_record[20], read_record[21], read_record[22],
                                read_record[23], read_record[24], read_record[25], read_record[26], read_record[27],
                                read_record[28], read_record[29]))
                            db.commit()  # 提交到数据库执行，一定要记提交哦
                        except Exception as e:
                            db.rollback()  # 发生错误时回滚
                            print(e)

                        cursor.close()
                        read_recordID += 1

            i += 1


if __name__ == "__main__":
    '''连接数据库'''
    db = pymysql.connect(host='127.0.0.1', port=3306,
                         user='root', passwd='1998msm322', db='nandflash_gan', charset='utf8')
    chip = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    testID = 5

    # start_line为下一次读取文件开始的行数，即该行还没有读取进数据库
    start_lines = 0
    # exit_line为下一次读取文件结束的行数（该行还没有读取进数据库）
    exit_lines = 40000000

    for item in chip:
        '''插入数据'''
        file_path = str(item).zfill(3) + ".log"
        chip = item
        read_file(data_path + file_path, chip, db, testID, start_lines, exit_lines)

    '''关闭连接'''
    db.close()
