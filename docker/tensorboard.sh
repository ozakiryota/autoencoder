#!/bin/bash

if [ $# != 1 ]; then
	echo "Usage: ./tensorboard.sh LOG_DIR"
	exit 1
fi
split_path=(${@//\// })
exp_dir=${split_path[-2]}
tb_dir=${split_path[-1]}

xhost +

image="autoencoder"
tag="latest"
exec_pwd=$(cd $(dirname $0); pwd)
home_dir="/home/user"

docker run \
	-it \
	--rm \
	-e local_uid=$(id -u $USER) \
	-e local_gid=$(id -g $USER) \
	-e "DISPLAY" \
	-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--net=host \
	-p 6006:6006 \
	-v $exec_pwd/../exp:$home_dir/$image/exp \
	$image:$tag \
	bash -c "tensorboard --logdir=$home_dir/$image/exp/$exp_dir/$tb_dir"