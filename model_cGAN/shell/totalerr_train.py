import os
command = "python ./model_cGAN/total_err_gen.py --train \
                           --test \
                           --cuda \
                           --latent_dim 20 \
                           --epochs 500 \
                           --batch_size 64 \
                           --err_data_name data.npy \
                           --condition_data_name condition.npy \
                           --save_model_epoch 50"

os.system(command)
