#!/bin/bash

exec_pwd=$(cd $(dirname $0); pwd)

cd $exec_pwd/../pyscr/exec

python3 rgbseg_evaluate.py \
    --dataset_dirs $HOME/dataset/airsim/tmp \
    --csv_name merged_file_list.csv \
    --csv_target_col 0 1 \
    --img_height 120 \
    --img_width 160 \
    --z_dim 5000 \
    --load_weights_dir $exec_pwd/../weights/rgbseg120pixel00001lre00001lrd1000sample64batch100epoch \
    --flag_show_reconstracted_images \
    --show_h 2 \
    --show_w 10
