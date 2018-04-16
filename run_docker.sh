#!/bin/bash


echo "NOTE: First build the docker container (only do this once for space though) './run_docker build'"
# Get argument for running the benchmark script.
ARG1=${1:-run}

if [ $ARG1 == "build" ]
then

sudo nvidia-docker build -t gpu .
    
fi

sudo nvidia-docker run -it \
    --mount type=bind,source="$(pwd)",target=/ML_OD_Benchmarking \
    gpu /bin/bash
