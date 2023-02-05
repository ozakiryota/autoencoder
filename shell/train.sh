#!/bin/bash

exec_pwd=$(cd $(dirname $0); pwd)

cd $exec_pwd/../pyscr/exec

python3 train.py \
    --dataset_dirs $HOME/dataset/airsim/center_road_south2north_1000 \
    --csv_target_col 1 \
    --img_height 120 \
    --img_width 160 \
    --batch_size 64 \
    --z_dim 3000 \
    --conv_unit_ch 64 \
    --load_weights_dir $exec_pwd/../weights/120pix3000z00001lre00001lrd1000sample64batch100epoch \
    --enc_lr 1e-4 \
    --dec_lr 1e-4 \
    --num_epochs 100 \
    --save_weights_step 100
