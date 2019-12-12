import os
command = "python ./model_cGAN/probability_distributions_gen.py --train \
                           --cuda \
                           --latent_dim 20 \
                           --epochs 500 \
                           --batch_size 64 \
                           --err_data_name data.npy \
                           --condition_data_name condition.npy \
                           --save_model_epoch 100"

os.system(command)
