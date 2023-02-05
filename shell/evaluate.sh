#!/bin/bash

exec_pwd=$(cd $(dirname $0); pwd)

cd $exec_pwd/../pyscr/exec

python3 evaluate.py \
    --dataset_dirs $HOME/dataset/airsim/tmp \
    --csv_target_col 0 \
    --img_height 120 \
    --img_width 160 \
    --z_dim 3000 \
    --conv_unit_ch 64 \
    --load_weights_dir $exec_pwd/../weights/120pix3000z00001lre00001lrd1000sample64batch100epoch \
    --flag_show_reconstracted_images
