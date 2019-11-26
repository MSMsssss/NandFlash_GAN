python ./model_cDCGAN2/main.py --train \
                           --cuda \
                           --ngf 128 \
                           --ndf 32 \
                           --latent_dim 100 \
                           --epochs 200 \
                           --batch_size 64 \
                           --err_data_name data.npy \
                           --condition_data_name condition.npy \
                           --save_model_epoch 20 \
