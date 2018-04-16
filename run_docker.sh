#!/bin/bash
sudo nvidia-docker build -t gpu .
sudo nvidia-docker run -it \
    --mount type=bind,source="$(pwd)",target=/ML_OD_Benchmarking \
    gpu /bin/bash
