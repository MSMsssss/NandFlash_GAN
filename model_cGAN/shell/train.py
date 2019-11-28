import os
command = "python ./model_cGAN/main.py --train \
                           --cuda \
                           --latent_dim 40 \
                           --epochs 500 \
                           --batch_size 64 \
                           --err_data_name data.npy \
                           --condition_data_name condition.npy \
                           --save_model_epoch 100"

os.system(command)
