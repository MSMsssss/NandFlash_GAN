import os

command = "python ./model_cGAN/total_err_gen.py --eval \
                           --cuda \
                           --latent_dim 5 \
                           --g_load_model_path totalerr_generator_epoch_%s.pth \
                           --d_load_model_path totalerr_discriminator_epoch_%s.pth \
                           --gen_start_pe 0 \
                           --gen_end_pe 17000 \
                           --gen_interval_pe 500 \
                           --generator_data_num 200"

for epoch in range(20, 220, 20):
    os.system(command % (epoch, epoch))
