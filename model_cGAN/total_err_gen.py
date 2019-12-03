import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)
sys.path.append(cur_path)

import argparse
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import torch.utils.data
from model_cGAN.config import Config
from model_cGAN.dataset import TotalErrDataset, TestDataSet
from data.connect_database import Connect, SqlConfig

config = Config()
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="训练模型")
parser.add_argument("--eval", action="store_true", help="运行模型")
# train与eval共用参数
parser.add_argument("--cuda", action="store_true", help="使用GPU")
parser.add_argument("--latent_dim", type=int, default=10, help="噪声维度")
# train参数
parser.add_argument("--lr", type=float, default=0.0002, help="学习速率")
parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=64, help="batch尺寸")
parser.add_argument("--save_model_epoch", type=int, default=20, help="设置每隔多少轮保存一次模型")
parser.add_argument("--err_data_name", default="", help="需保存在./download_data下，为空时从数据库读取")
parser.add_argument("--condition_data_name", default="", help="需保存在./download_data下，为空时从数据库读取")
parser.add_argument("--test", action="store_true", help="使用测试模式数据集")
# eval参数
parser.add_argument("--g_load_model_path", default="",
                    help="生成器模型参数保存文件名，必须放置在同目录的save_model文件夹下，如msm.pth")
parser.add_argument("--d_load_model_path", default="",
                    help="判别器模型参数保存文件名，必须放置在同目录的save_model文件夹下，如msm.pth")
parser.add_argument("--gen_start_pe", type=int, default=0, help="生成假数据的开始pe")
parser.add_argument("--gen_end_pe", type=int, default=17000, help="生成假数据的结束pe")
parser.add_argument("--gen_interval_pe", type=int, default=500, help="生成假数据的间隔pe")
parser.add_argument("--generator_data_num", type=int, default=200,
                    help="每个pe生成generator_data_num个数据")
opt = parser.parse_args()
print(opt)


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
            *block(opt.latent_dim + config.condition_dim, 512, normalize=False),
            *block(512, 2048),
            *block(2048, 4096),
            *block(4096, 2048),
            nn.Linear(2048, config.g_output_dim),
        )

    def forward(self, noise, condition):
        gen_input = torch.cat((condition, noise), 1)
        fake_data = self.model(gen_input)
        return fake_data


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(config.g_output_dim + config.condition_dim, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 2048),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, err_data, condition):
        d_in = torch.cat((err_data, condition), 1)
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
loss_function = nn.MSELoss()


def load_model(g_model_path, d_model_path):
    generator.load_state_dict(torch.load(g_model_path))
    discriminator.load_state_dict(torch.load(d_model_path))


# 训练模型
def train():
    # 初始化数据集
    print("加载数据中...")
    if opt.test:
        real_data_set = TestDataSet()
    else:
        if opt.err_data_name != "":
            real_data_set = TotalErrDataset(err_data_path=cur_path + "/download_data/" + opt.err_data_name,
                                            condition_data_path=cur_path + "/download_data/" + opt.condition_data_name,
                                            normalize=True)
        else:
            real_data_set = TotalErrDataset(normalize=True)
    real_data_loader = torch.utils.data.DataLoader(dataset=real_data_set, batch_size=opt.batch_size, shuffle=True)
    print('数据加载完成，块数据:%s条' % len(real_data_set))

    generator.train()
    discriminator.train()

    real_label = 1
    fake_label = 0

    if opt.test:
        condition_set = [0.001, 0.01, 0.1, 1, 10]
    else:
        condition_set = [((x / config.max_pe) - 0.5) / 0.5 for x in config.pe_set]

    for epoch in range(opt.epochs):
        for i, (err_data, condition) in enumerate(real_data_loader):
            batch_size = err_data.shape[0]

            # 真实数据
            real_err_data = err_data.to(device)
            real_condition = condition.to(device)

            # ---------------------
            #  训练分类器
            # ---------------------

            optimizer_D.zero_grad()

            # 训练真实数据
            label = torch.full((batch_size,), real_label, device=device)
            output = discriminator(real_err_data, real_condition)
            d_real = output.mean().item()

            # 计算损失
            lossD_real = loss_function(output.squeeze(), label)
            lossD_real.backward()

            # 生成噪音和标签
            # 噪声采样和假数据条件生成
            z = torch.randn(batch_size, opt.latent_dim, device=device)
            gen_condition = torch.from_numpy(np.random.choice(
                condition_set, (batch_size, config.condition_dim))).to(device=device, dtype=torch.float32)

            # 训练生成数据
            label.fill_(fake_label)
            fake_err_data = generator(z, gen_condition)
            output = discriminator(fake_err_data.detach(), gen_condition)
            d_fake1 = output.mean().item()

            # 计算损失
            lossD_fake = loss_function(output.squeeze(), label)
            lossD_fake.backward()

            # 总损失
            d_loss = (lossD_fake + lossD_real) / 2
            optimizer_D.step()

            # -----------------
            #  训练生成器
            # -----------------
            optimizer_G.zero_grad()

            # 计算生成器损失
            label.fill_(real_label)
            output = discriminator(fake_err_data, gen_condition)
            d_fake2 = output.mean().item()
            g_loss = loss_function(output.squeeze(), label)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f] [D1(z): %f] [D2(z): %f]"
                % (epoch + 1, opt.epochs, i, len(real_data_loader),
                   d_loss.item(), g_loss.item(), d_real, d_fake1, d_fake2)
            )

        if (epoch + 1) % opt.save_model_epoch == 0:
            torch.save(generator.state_dict(), "%s/totalerr_generator_epoch_%s.pth" % (config.model_saved_path, epoch + 1))
            torch.save(discriminator.state_dict(), "%s/totalerr_discriminator_epoch_%s.pth" %
                       (config.model_saved_path, epoch + 1))


def model_eval():
    connect = Connect(SqlConfig.generator_database)
    generator.eval()
    discriminator.eval()

    generator.requires_grad_(False)
    discriminator.requires_grad_(False)

    gen_data_set = []
    condition_set = []
    for pe in range(opt.gen_start_pe, opt.gen_end_pe, opt.gen_interval_pe):
        z = torch.randn(opt.generator_data_num, opt.latent_dim, device=device)

        # 生成假数据
        condition = torch.ones((opt.generator_data_num, config.condition_dim),
                               device=device, requires_grad=False).fill_(((pe / config.max_pe) - 0.5) / 0.5)

        gen_err_data = generator(z, condition).squeeze()
        gen_data_set.append(gen_err_data.detach().cpu())
        condition_set.append(torch.zeros((opt.generator_data_num, config.condition_dim),
                                         dtype=torch.int32).fill_(pe).detach().cpu())
        print("pe: %s is done" % pe)
    print("all is done")

    s = opt.g_load_model_path
    epoch = int(s[s.rfind("_") + 1:s.rfind(".")])

    np.save(cur_path + "/gen_data/totalerr_gen_data_%s.npy" % epoch, torch.cat(gen_data_set, 0).numpy())
    np.save(cur_path + "/gen_data/totalerr_gen_condition_%s.npy" % epoch, torch.cat(condition_set, 0).numpy())


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
