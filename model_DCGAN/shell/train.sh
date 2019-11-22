python ./model_DCGAN/main.py --train --epochs 500 \
                           --batch_size 64 \
                           --cuda \
                           --err_data_name data.npy \
                           --condition_data_name condition.npy \
                           --save_model_epoch 100 \
                           --ngf 32 \
                           --ndf 4 \
                           --latent_dim 100 \
