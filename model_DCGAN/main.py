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
from model_DCGAN.config import Config
import torch.utils.data
from data.dataset import Dataset, TestDataset
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
parser.add_argument("--lr", type=float, default=0.0002, help="学习速率")
parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=64, help="batch尺寸")
parser.add_argument("--save_model_epoch", type=int, default=50, help="设置每隔多少轮保存一次模型")
parser.add_argument("--gen_start_pe", type=int, default=0, help="生成假数据的开始pe")
parser.add_argument("--gen_end_pe", type=int, default=17000, help="生成假数据的结束pe")
parser.add_argument("--gen_interval_pe", type=int, default=500, help="生成假数据的间隔pe")
parser.add_argument("--generator_data_num", type=int, default=200,
                    help="每个pe生成generator_data_num个数据")
parser.add_argument("--err_data_name", default="", help="需保存在./data/download_data下，为空时从数据库读取")
parser.add_argument("--condition_data_name", default="", help="需保存在./data/download_data下，为空时从数据库读取")
parser.add_argument("--test", action="store_true", help="测试模式")
parser.add_argument("--ngf", type=int, default=32, help="生成器基准通道数")
parser.add_argument("--ndf", type=int, default=4, help="分类器基准通道数")
parser.add_argument("--latent_dim", type=int, default=100, help="噪声维度")
opt = parser.parse_args()
print(opt)


# 模型权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


ngf = opt.ngf
ndf = opt.ndf
# err_data为2304 * 16只有一个channel
nc = 1


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_module = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(opt.latent_dim + config.condition_dim, ngf * 16, (4, 2), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 2
            nn.ConvTranspose2d(ngf * 16, ngf * 8, (4, 2), (4, 2), (0, 0), bias=False, output_padding=(1, 0)),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 17 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, (4, 2), (4, 2), (0, 0), bias=False, dilation=(2, 1), output_padding=(1, 0)),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 72 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, (4, 2), (4, 2), (0, 0), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 288 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, (4, 3), (4, 1), (0, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 1152 x 16
            nn.ConvTranspose2d(ngf, nc, (4, 3), (2, 1), (1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 2304 x 16
        )

    def forward(self, noise, condition):
        input = torch.cat((noise, condition.view(-1, config.condition_dim, 1, 1)), 1)
        output = self.conv_module(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.condition_channel = torch.ones((opt.batch_size, config.condition_dim, config.height, config.width),
                                            dtype=torch.float32, device=device, requires_grad=False)
        self.conv_module = nn.Sequential(
            # input is (nc) x 2304 x 16
            nn.Conv2d(nc + config.condition_dim, ndf, (4, 3), (2, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf)
            nn.Conv2d(ndf, ndf * 2, (4, 3), (4, 1), (0, 1), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2)
            nn.Conv2d(ndf * 2, ndf * 4, (4, 2), (4, 2), (0, 0), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4)
            nn.Conv2d(ndf * 4, ndf * 8, (4, 2), (4, 2), (0, 0), bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4)
            nn.Conv2d(ndf * 8, ndf * 16, (4, 2), (4, 2), (0, 0), bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8)
            nn.Conv2d(ndf * 16, 1, (4, 2), (1, 1), (0, 0), bias=False),
            nn.Sigmoid()
        )

    def forward(self, err_data, condition):
        batch_size = err_data.shape[0]
        for i in range(batch_size):
            for j in range(config.condition_dim):
                self.condition_channel[i][j].fill_(condition[i][j])

        input = torch.cat((err_data, self.condition_channel[:batch_size]), 1)
        return self.conv_module(input)


# 设备
device = torch.device("cuda:0" if opt.cuda else "cpu")

# 初始化生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

# 初始化优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=config.betas)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=config.betas)

# 初始化损失函数
loss_function = nn.BCELoss()


def load_model(g_model_path, d_model_path):
    generator.load_state_dict(torch.load(g_model_path))
    discriminator.load_state_dict(torch.load(d_model_path))


# 训练模型
def train():
    # 初始化数据集
    print("加载数据中...")
    if opt.test:
        real_data_set = TestDataset()
    else:
        if opt.err_data_name != "":
            real_data_set = Dataset(err_data_path=root_path + "/data/download_data/" + opt.err_data_name,
                                    condition_data_path=root_path + "/data/download_data/" + opt.condition_data_name)
        else:
            real_data_set = Dataset()
    real_data_loader = torch.utils.data.DataLoader(dataset=real_data_set, batch_size=opt.batch_size, shuffle=True)
    print('数据加载完成，块数据:%s条' % len(real_data_set))

    generator.train()
    discriminator.train()

    real_label = 1
    fake_label = 0

    condition_set = [((x / config.max_pe) - 0.5) / 0.5 for x in config.pe_set]

    for epoch in range(opt.epochs):
        for i, (err_data, condition) in enumerate(real_data_loader):
            batch_size = err_data.shape[0]

            # 真实数据
            real_err_data = err_data.view(batch_size, 1, config.height, config.width).to(device)
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
            z = torch.randn(batch_size, opt.latent_dim, 1, 1, device=device)
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
            torch.save(generator.state_dict(), "%s/generator_epoch_%s.pth" % (config.model_saved_path, epoch + 1))
            torch.save(discriminator.state_dict(), "%s/discriminator_epoch_%s.pth" %
                       (config.model_saved_path, epoch + 1))


def model_eval():
    connect = Connect(SqlConfig.generator_database)
    generator.eval()
    gen_data_set = []
    for pe in range(opt.gen_start_pe, opt.gen_end_pe, opt.gen_interval_pe):
        z = torch.randn(opt.generator_data_num, opt.latent_dim, 1, 1, device=device)

        # 生成假数据
        condition = torch.ones((opt.generator_data_num, config.condition_dim),
                               device=device, requires_grad=False).fill_(((pe / config.max_pe) - 0.5) / 0.5)

        gen_err_data = generator(z, condition).squeeze()
        gen_data_set.append(gen_err_data.cpu())

    s = opt.g_load_model_path
    epoch = int(s[s.rfind("_") + 1:s.rfind(".")])

    np.save(cur_path + "/gen_data/gen_data_%s.npy" % epoch, torch.cat(gen_data_set, 0).numpy())
    np.save(cur_path + "/gen_data/condition_%s.npy" % epoch,
            np.array([opt.gen_start_pe, opt.gen_end_pe, opt.gen_interval_pe], dtype=np.int))


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
