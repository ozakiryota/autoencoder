#!/bin/bash

exec_pwd=$(cd $(dirname $0); pwd)

cd $exec_pwd/../pyscr/exec

python3 evaluate.py \
    --dataset_dirs $HOME/dataset/airsim/tmp \
    --csv_target_col 0 \
    --img_height 120 \
    --img_width 160 \
    --z_dim 5000 \
    --deconv_unit_ch 64 \
    --load_exp_dir $exec_pwd/../exp/120pix5000z64deconv00001lre00001lrd1000sample64batch100epoch/100epoch \
    --flag_show_reconstracted_images
