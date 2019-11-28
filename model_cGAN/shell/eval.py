import os

command = "python ./model_cGAN/main.py --eval \
                           --cuda \
                           --latent_dim 20 \
                           --g_load_model_path generator_epoch_%s.pth \
                           --d_load_model_path discriminator_epoch_%s.pth \
                           --gen_start_pe 0 \
                           --gen_end_pe 17000 \
                           --gen_interval_pe 500 \
                           --generator_data_num 200"

for epoch in range(20, 220, 20):
    os.system(command % (epoch, epoch))