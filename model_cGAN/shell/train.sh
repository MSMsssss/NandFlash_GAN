python ./model_cGAN/main.py --train \
                           --cuda \
                           --latent_dim 20 \
                           --epochs 200 \
                           --batch_size 64 \
                           --err_data_name data.npy \
                           --condition_data_name condition.npy \
                           --save_model_epoch 20 \
