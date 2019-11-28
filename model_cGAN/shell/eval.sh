epoch=20
while $epoch <= 200
do
python ./model_cGAN/main.py --eval \
                           --cuda \
                           --latent_dim 20 \
                           --g_load_model_path generator_epoch_100.pth \
                           --d_load_model_path discriminator_epoch_100.pth \
                           --gen_start_pe 0 \
                           --gen_end_pe 17000 \
                           --gen_interval_pe 500 \
                           --generator_data_num 20
epoch=$epoch+20
done


