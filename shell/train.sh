#!/bin/bash

exec_pwd=$(cd $(dirname $0); pwd)

cd $exec_pwd/../pyscr/exec

python3 train.py \
    --dataset_dirs $HOME/dataset/airsim/center_road_south2north_1000 \
    --csv_target_col 1 \
    --img_height 120 \
    --img_width 160 \
    --batch_size 128 \
    --z_dim 2000 \
    --enc_lr 1e-4 \
    --dec_lr 1e-4 \
    --num_epochs 100 \
    --save_weights_step 100
