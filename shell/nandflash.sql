/*
 Navicat Premium Data Transfer

 Source Server         : mysql
 Source Server Type    : MySQL
 Source Server Version : 80016
 Source Host           : 10.147.20.88:3306
 Source Schema         : nandflash

 Target Server Type    : MySQL
 Target Server Version : 80016
 File Encoding         : 65001

 Date: 02/12/2019 15:05:47
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for blockbe
-- ----------------------------
DROP TABLE IF EXISTS `blockbe`;
CREATE TABLE `blockbe`  (
  `recordID` bigint(64) NOT NULL AUTO_INCREMENT,
  `testID` int(16) NOT NULL,
  `pe` int(16) NOT NULL,
  `rt` int(15) NOT NULL,
  `rd` int(24) NOT NULL,
  `chip` int(10) NOT NULL,
  `ce` int(3) NOT NULL,
  `die` int(3) NOT NULL,
  `block` int(15) NOT NULL,
  `err` int(13) NOT NULL,
  `recordtime` timestamp(0) DEFAULT NULL,
  PRIMARY KEY (`recordID`) USING BTREE,
  INDEX `testID1`(`testID`) USING BTREE,
  INDEX `pe`(`pe`) USING BTREE,
  INDEX `rt`(`rt`) USING BTREE,
  INDEX `rd`(`rd`) USING BTREE,
  INDEX `chip`(`chip`) USING BTREE,
  INDEX `ce`(`ce`) USING BTREE,
  INDEX `block`(`block`) USING BTREE,
  CONSTRAINT `testID` FOREIGN KEY (`testID`) REFERENCES `testconfig` (`testID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 865053 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for blockstatus
-- ----------------------------
DROP TABLE IF EXISTS `blockstatus`;
CREATE TABLE `blockstatus`  (
  `blockID` varchar(11) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `modelID` int(2) NOT NULL,
  `chip` int(10) NOT NULL,
  `ce` int(3) NOT NULL,
  `die` int(3) NOT NULL,
  `block` int(15) NOT NULL,
  `pe` int(16) NOT NULL,
  `health` int(1) NOT NULL,
  `recordtime` timestamp(0) DEFAULT NULL,
  PRIMARY KEY (`blockID`) USING BTREE,
  INDEX `modelID1`(`modelID`) USING BTREE,
  CONSTRAINT `modelID1` FOREIGN KEY (`modelID`) REFERENCES `nandmodel` (`modelID`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for nandmodel
-- ----------------------------
DROP TABLE IF EXISTS `nandmodel`;
CREATE TABLE `nandmodel`  (
  `modelID` int(2) NOT NULL AUTO_INCREMENT,
  `modelname` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `modelname_abbr` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `type` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `chipnum_min` int(3) NOT NULL,
  `chipnum_max` int(3) NOT NULL,
  `ce` int(2) NOT NULL,
  `die` int(2) NOT NULL,
  `plane` int(2) NOT NULL,
  `block` int(5) NOT NULL,
  `page` int(6) NOT NULL,
  `pagesize` int(10) NOT NULL,
  PRIMARY KEY (`modelID`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for terase
-- ----------------------------
DROP TABLE IF EXISTS `terase`;
CREATE TABLE `terase`  (
  `recordID` int(64) NOT NULL AUTO_INCREMENT,
  `testID` int(16) NOT NULL,
  `pe` int(16) NOT NULL,
  `chip` int(10) NOT NULL,
  `ce` int(3) NOT NULL,
  `die` int(3) NOT NULL,
  `block` int(15) NOT NULL,
  `terase` int(15) NOT NULL,
  `recordtime` timestamp(0) DEFAULT NULL,
  PRIMARY KEY (`recordID`) USING BTREE,
  INDEX `pe`(`pe`) USING BTREE,
  INDEX `chip`(`chip`) USING BTREE,
  INDEX `ce`(`ce`) USING BTREE,
  INDEX `block`(`block`) USING BTREE,
  INDEX `testID_erase`(`testID`) USING BTREE,
  CONSTRAINT `testID_erase` FOREIGN KEY (`testID`) REFERENCES `testconfig` (`testID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 21772455 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for testconfig
-- ----------------------------
DROP TABLE IF EXISTS `testconfig`;
CREATE TABLE `testconfig`  (
  `testID` int(16) NOT NULL,
  `modelID` int(2) NOT NULL,
  `cmd` varchar(2048) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `admin` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  PRIMARY KEY (`testID`) USING BTREE,
  INDEX `modelID2`(`modelID`) USING BTREE,
  CONSTRAINT `modelID2` FOREIGN KEY (`modelID`) REFERENCES `nandmodel` (`modelID`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for testgroup
-- ----------------------------
DROP TABLE IF EXISTS `testgroup`;
CREATE TABLE `testgroup`  (
  `groupID` int(16) NOT NULL,
  `testID` int(16) NOT NULL,
  `chip` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `ce` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `die` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `block` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `description` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  PRIMARY KEY (`groupID`) USING BTREE,
  INDEX `testID_9`(`testID`) USING BTREE,
  CONSTRAINT `testID_9` FOREIGN KEY (`testID`) REFERENCES `testconfig` (`testID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for tprog
-- ----------------------------
DROP TABLE IF EXISTS `tprog`;
CREATE TABLE `tprog`  (
  `recordID` bigint(64) NOT NULL AUTO_INCREMENT,
  `testID` int(8) NOT NULL,
  `pe` int(16) NOT NULL,
  `chip` int(10) NOT NULL,
  `ce` int(3) NOT NULL,
  `die` int(3) NOT NULL,
  `block` int(15) NOT NULL,
  `page` int(10) NOT NULL,
  `pagetype` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `tprog` int(15) NOT NULL,
  `recordtime` timestamp(0) DEFAULT NULL,
  PRIMARY KEY (`recordID`) USING BTREE,
  INDEX `pe`(`pe`) USING BTREE,
  INDEX `chip`(`chip`) USING BTREE,
  INDEX `ce`(`ce`) USING BTREE,
  INDEX `block`(`block`) USING BTREE,
  INDEX `page`(`page`) USING BTREE,
  INDEX `testID_prog`(`testID`) USING BTREE,
  CONSTRAINT `testID_prog` FOREIGN KEY (`testID`) REFERENCES `testconfig` (`testID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 468385570 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for tread
-- ----------------------------
DROP TABLE IF EXISTS `tread`;
CREATE TABLE `tread`  (
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
  INDEX `synthetic`(`testID`, `pe`, `chip`, `ce`, `die`, `block`) USING BTREE,
  CONSTRAINT `testID_read` FOREIGN KEY (`testID`) REFERENCES `testconfig` (`testID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE = InnoDB AUTO_INCREMENT = 468335100 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
