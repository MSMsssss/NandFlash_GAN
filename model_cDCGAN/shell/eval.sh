python ./model_cDCGAN/main.py --eval \
                           --cuda \
                           --ngf 128 \
                           --ndf 32 \
                           --latent_dim 100 \
                           --g_load_model_path generator_epoch_100.pth \
                           --d_load_model_path discriminator_epoch_100.pth \
                           --gen_start_pe 0 \
                           --gen_end_pe 17000 \
                           --gen_interval_pe 500 \
                           --generator_data_num 200


