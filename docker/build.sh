#!/bin/bash

image="autoencoder"
tag="latest"

docker build . \
    -t $image:$tag \
    --build-arg cache_bust=$(date +%s)
