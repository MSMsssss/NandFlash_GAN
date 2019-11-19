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