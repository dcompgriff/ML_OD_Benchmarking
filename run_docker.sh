#!/bin/bash


echo "NOTE: First build the docker container (only do this once for space though) './run_docker build'"
# Get argument for running the benchmark script.
ARG1=${1:-run}

if [ $ARG1 == "build" ]
then

sudo nvidia-docker build -t gpu .
    
fi

# If detectron-download-cache doesn't exist, then make it.
if [ -d "$(pwd)/detectron-download-cache/" ]
then
    echo "detectron-download-cache/ folder exists."
else
    echo "detectron-download-cache/ folder missing! Creating..."
    mkdir "$(pwd)/detectron-download-cache/"
fi

# If detectron model cache folder is empty
if [ -z "$(ls -A $(pwd)/detectron-download-cache/)" ]; then
    echo "$(pwd)/detectron-download-cache/ is empty!"
    echo "Calling $(pwd)/detectron_scripts/cache.sh to download all model .pkl files to cache..."
    ./detectron_scripts/cache.sh
else
    echo "$(pwd)/detectron-download-cache/ NOT empty, so no .pkl models will be downloaded."
fi

sudo nvidia-docker run -it \
     --mount type=bind,source="$(pwd)",target=/ML_OD_Benchmarking \
    gpu /bin/bash
