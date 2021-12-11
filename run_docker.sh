#!/bin/bash
HOST_MODELS="$HOME/vqganclip/models"
HOST_OUTPUT="$HOME/vqganclip/outputs"
HOST_PORT=8888
sudo docker run --name vqganclip --gpus all --rm -it -p $HOST_PORT:8888 \
        -v "$(pwd):/tf/src" \
        -v "$HOST_MODELS:/tf/models" \
        -v "$HOST_OUTPUT:/tf/outputs" \
        vgan/vqganclip:latest\
        bash
