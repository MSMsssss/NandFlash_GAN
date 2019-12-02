CREATE DATABASE IF NOT EXISTS NandFlash_GAN;
USE NandFlash_GAN;

CREATE TABLE IF NOT EXISTS blocks(
   block_id INT UNSIGNED AUTO_INCREMENT, -- block编号
   pe INT NOT NULL, -- 生成该块时的pe次数
   total_err INT NOT NULL, -- 该块的总错误次数
   submit_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, -- 数据生成时间
   PRIMARY KEY (block_id)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS generator_data(
   block_id INT UNSIGNED NOT NULL,
   page_id INT NOT NULL,
   err INT NOT NULL,
   f0 INT NOT NULL,
   f1 INT NOT NULL,
   f2 INT NOT NULL,
   f3 INT NOT NULL,
   f4 INT NOT NULL,
   f5 INT NOT NULL,
   f6 INT NOT NULL,
   f7 INT NOT NULL,
   f8 INT NOT NULL,
   f9 INT NOT NULL,
   f10 INT NOT NULL,
   f11 INT NOT NULL,
   f12 INT NOT NULL,
   f13 INT NOT NULL,
   f14 INT NOT NULL,
   f15 INT NOT NULL,
   PRIMARY KEY (block_id, page_id),
   FOREIGN KEY (block_id) REFERENCES blocks(block_id)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS real_blocks(
   block_id INT UNSIGNED AUTO_INCREMENT, -- block编号
   pe INT NOT NULL, -- 测试该块时的pe次数
   total_err INT NOT NULL, -- 该块的总错误次数
   submit_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, -- 数据生成时间
   PRIMARY KEY (block_id)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;


CREATE TABLE IF NOT EXISTS `tread`  (
  `recordID` bigint(64) NOT NULL AUTO_INCREMENT,
  `testID` int(16) NOT NULL,
  `pe` int(16) NOT NULL,
  `rt` int(15) NOT NULL,
  `rd` int(24) NOT NULL,
  `chip` int(10) NOT NULL,
  `ce` int(3) NOT NULL,
  `die` int(3) NOT NULL,
  `block` int(15) NOT NULL,
  `page` int(10) NOT NULL,
  `pagetype` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `tread` int(10) NOT NULL,
  `err` int(13) NOT NULL,
  `f0` int(13) NOT NULL,
  `f1` int(13) NOT NULL,
  `f2` int(13) NOT NULL,
  `f3` int(13) NOT NULL,
  `f4` int(13) NOT NULL,
  `f5` int(13) NOT NULL,
  `f6` int(13) NOT NULL,
  `f7` int(13) NOT NULL,
  `f8` int(13) NOT NULL,
  `f9` int(13) NOT NULL,
  `f10` int(13) NOT NULL,
  `f11` int(13) NOT NULL,
  `f12` int(13) NOT NULL,
  `f13` int(13) NOT NULL,
  `f14` int(13) NOT NULL,
  `f15` int(13) NOT NULL,
  `recordtime` timestamp(0) DEFAULT NULL,
  PRIMARY KEY (`recordID`) USING BTREE,
  INDEX `pe`(`pe`) USING BTREE,
  INDEX `chip`(`chip`) USING BTREE,
  INDEX `ce`(`ce`) USING BTREE,
  INDEX `block`(`block`) USING BTREE,
  INDEX `rt`(`rt`) USING BTREE,
  INDEX `rd`(`rd`) USING BTREE,
  INDEX `page`(`page`) USING BTREE,
  INDEX `die`(`die`) USING BTREE,
  INDEX `synthetic`(`testID`, `pe`, `chip`, `ce`, `die`, `block`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 468335100 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;


CREATE TABLE IF NOT EXISTS `testgroup`(
  `groupID` int(16) NOT NULL,
  `testID` int(16) NOT NULL,
  `chip` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `ce` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `die` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `block` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `description` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  PRIMARY KEY (`groupID`) USING BTREE,
  INDEX `testID_9`(`testID`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;