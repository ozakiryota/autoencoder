#!/bin/bash

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
	--gpus all \
	-v $HOME/dataset:$home_dir/dataset \
	-v $(pwd)/../log:$home_dir/$image/log \
	-v $(pwd)/../weights:$home_dir/$image/weights \
	-v $(pwd)/../fig:$home_dir/$image/fig \
	-v $(pwd)/../pyscr:$home_dir/$image/pyscr \
	-v $(pwd)/../shell:$home_dir/$image/shell \
	$image:$tag