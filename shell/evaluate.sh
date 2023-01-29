#!/bin/bash

exec_pwd=$(cd $(dirname $0); pwd)

cd $exec_pwd/../pyscr/exec

python3 evaluate.py \
    --dataset_dirs $HOME/dataset/airsim/center_road_south2north_1000 \
    --csv_target_col 1 \
    --img_height 120 \
    --img_width 160 \
    --load_weights_dir $exec_pwd/../weights/120pixel00001lre00001lrd1000sample128batch100epoch \
    --flag_show_reconstracted_images
