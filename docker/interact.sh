#!/bin/bash

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
	--gpus all \
	-v $HOME/dataset:$home_dir/dataset \
	-v $exec_pwd/../exp:$home_dir/$image/exp \
	-v $exec_pwd/../pyscr:$home_dir/$image/pyscr \
	-v $exec_pwd/../shell:$home_dir/$image/shell \
	$image:$tag