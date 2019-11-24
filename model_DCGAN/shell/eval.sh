python ./model_DCGAN/main.py --eval \
                           --cuda \
                           --g_load_model_path generator_epoch_20.pth \
                           --d_load_model_path discriminator_epoch_20.pth \
                           --gen_start_pe 0 \
                           --gen_end_pe 17000 \
                           --gen_interval_pe 500

