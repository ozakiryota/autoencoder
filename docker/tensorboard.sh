#!/bin/bash

if [ $# != 1 ]; then
	echo "Usage: ./tensorboard.sh LOG_DIR"
	exit 1
fi
log_dir=`basename $@`

xhost +

image="autoencoder"
tag="latest"
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
	-v $(pwd)/../log:$home_dir/$image/log \
	$image:$tag \
	bash -c "tensorboard --logdir=$home_dir/$image/log/$log_dir"