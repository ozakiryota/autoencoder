#!/bin/bash

exec_pwd=$(cd $(dirname $0); pwd)

cd $exec_pwd/../pyscr/exec

python3 rgbseg_train.py \
    --dataset_dirs $HOME/dataset/airsim/center_road_south2north_1000 \
    --csv_name file_list_grayscaled.csv \
    --csv_target_col 0 1 \
    --img_height 120 \
    --img_width 160 \
    --rotation_range_deg 5 \
    --batch_size 32 \
    --z_dim 5000 \
    --deconv_unit_ch 128 \
    --conv_unit_ch 4 \
    --enc_lr 1e-4 \
    --dec_lr 1e-4 \
    --num_epochs 100 \
    --save_weights_step 10
