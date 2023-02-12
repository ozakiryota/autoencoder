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
    --deconv_unit_ch 128 \
    --conv_unit_ch 4 \
    --load_exp_dir $exec_pwd/../exp/rgbseg120pix5000z4conv128deconv00001lre00001lrd1000sample32batch100epoch/100epoch \
    --flag_show_reconstracted_images
