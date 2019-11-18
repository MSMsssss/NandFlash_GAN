import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
sys.path.append(root_path + "\\model_DNN")
sys.path.append(root_path + "\\data")

import argparse
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from model_DNN.config import Config
import torch.utils.data
from data.dataset import Dataset
from data.connect_database import Connect, SqlConfig

config = Config()
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="训练模型")
parser.add_argument("--eval", action="store_true", help="运行模型")
parser.add_argument("--g_load_model_path", default="",
                    help="生成器模型参数保存文件名，必须放置在同目录的save_model文件夹下，如msm.pth")
parser.add_argument("--d_load_model_path", default="",
                    help="判别器模型参数保存文件名，必须放置在同目录的save_model文件夹下，如msm.pth")
parser.add_argument("--cuda", action="store_true", help="使用GPU训练")
parser.add_argument("--lr", type=float, default=0.02, help="学习速率")
parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=32, help="batch尺寸")
parser.add_argument("--save_model_epoch", type=int, default=50, help="设置每隔多少轮保存一次模型")
parser.add_argument("--gen_start_pe", type=int, default=0, help="生成假数据的开始pe")
parser.add_argument("--gen_end_pe", type=int, default=15000, help="生成假数据的结束pe")
parser.add_argument("--gen_interval_pe", type=int, default=1000, help="生成假数据的间隔pe")
parser.add_argument("--generator_data_num", type=int, default=1,
                    help="每个pe生成generator_data_num个数据")
opt = parser.parse_args()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(config.latent_dim + config.condition_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, config.width * config.height),
        )

    def forward(self, noise, condition):
        # Concatenate label embedding and image to produce input
        condition = nn.BatchNorm1d(condition)
        gen_input = torch.cat((condition, noise), -1)
        err_data = self.model(gen_input)
        err_data = err_data.view(err_data.size(0), config.height, config.width)
        return err_data


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(config.condition_dim + config.width * config.height, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, err_data, condition):
        # Concatenate label embedding and image to produce input
        condition = nn.BatchNorm1d(condition)
        d_in = torch.cat((err_data.view(err_data.size(0), -1), condition), -1)
        validity = self.model(d_in)
        return validity


# 设备
device = torch.device("cuda:0" if opt.cuda else "cpu")

# 初始化生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 初始化优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=config.betas)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=config.betas)

# 初始化损失函数
loss_function = nn.MSELoss().to(device)


def load_model(g_model_path, d_model_path):
    generator.load_state_dict(torch.load(g_model_path))
    discriminator.load_state_dict(torch.load(d_model_path))


# 训练模型
def train():
    # 初始化数据集
    print("加载数据中...")
    real_data_set = Dataset()
    real_data_loader = torch.utils.data.DataLoader(dataset=real_data_set, batch_size=opt.batch_size, shuffle=True)
    print('数据加载完成')

    generator.train()
    discriminator.train()

    for epoch in range(opt.epochs):
        for i, (err_data, condition) in enumerate(real_data_loader):
            batch_size = err_data.shape[0]

            # 初始化标签
            valid = torch.ones((batch_size, 1), dtype=torch.float32, requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), dtype=torch.float32, requires_grad=False).to(device)

            # 真实数据
            real_err_data = err_data.to(device)
            real_err_data.requires_grad = False
            real_condition = condition.to(device)

            # -----------------
            #  训练生成器
            # -----------------
            optimizer_G.zero_grad()

            # 噪声采样和假数据条件生成
            z = torch.randn((batch_size, config.latent_dim), requires_grad=False).to(device=device, dtype=torch.float32)
            gen_condition = torch.from_numpy(np.random.choice(
                config.pe_set, (batch_size, config.condition_dim))).to(device=device, dtype=torch.float32)

            gen_condition.requires_grad = False

            # 生成假数据
            gen_err_data = generator(z, gen_condition)

            # 计算生成器损失
            validity = discriminator(gen_err_data, gen_condition)
            g_loss = loss_function(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  训练分类器
            # ---------------------

            optimizer_D.zero_grad()

            # 计算分类器对真实数据的损失
            validity_real = discriminator(real_err_data, real_condition)
            d_real_loss = loss_function(validity_real, valid)

            # 计算分类器对生成数据的损失
            validity_fake = discriminator(gen_err_data.detach(), gen_condition)
            d_fake_loss = loss_function(validity_fake, fake)

            # 计算总损失
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch + 1, opt.epochs, i, len(real_data_loader), d_loss.item(), g_loss.item())
            )
        if (epoch + 1) % opt.save_model_epoch == 0:
            torch.save(generator.state_dict(), "%s/generator_epoch_%s.pth" % (config.model_saved_path, epoch + 1))
            torch.save(discriminator.state_dict(), "%s/discriminator_epoch_%s.pth" %
                       (config.model_saved_path, epoch + 1))


def model_eval():
    connect = Connect(SqlConfig.generator_database)
    generator.eval()
    for pe in range(opt.gen_start_pe, opt.gen_end_pe, opt.gen_interval_pe):
        for i in range(opt.generator_data_num):
            z = torch.randn((1, config.latent_dim), requires_grad=False).to(device)

            # 生成假数据
            gen_err_data = generator(z, torch.tensor([[pe]], dtype=torch.float32).to(device))
            connect.insert_block_data(gen_err_data, pe)
            print("(序号: %s, pe: %s) fake block已导入数据库" % (i, pe))


def run():
    if opt.train and opt.eval:
        print("error")
    elif opt.train:
        train()
    elif opt.eval:
        load_model(config.model_saved_path + opt.g_load_model_path, config.model_saved_path + opt.d_load_model_path)
        print("模型加载完成")
        model_eval()


if __name__ == "__main__":
    run()
