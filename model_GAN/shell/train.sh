python ./model_GAN/main.py --train --epochs 200 \
                           --batch_size 64 \
                           --cuda \
                           --err_data_name data.npy \
                           --condition_data_name condition.npy \
                           --save_model_epoch 20

