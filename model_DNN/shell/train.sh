python ./model_DNN/main.py --train --epochs 500 \
                           --batch_size 64 \
                           --cuda \
                           --err_data_path data.npy \
                           --condition_data_path condition.npy

