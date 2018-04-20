#!/bin/bash

echo "Starting sequential image transform process."
start=`date +%s`

python2 image_transformer.py --output-dir ../data/inputs/transformed/ --image-ext jpg --replace-output-dir 0 ../data/inputs/input0 &
python2 image_transformer.py --output-dir ../data/inputs/transformed/ --image-ext jpg --replace-output-dir 0 ../data/inputs/input1 &
python2 image_transformer.py --output-dir ../data/inputs/transformed/ --image-ext jpg --replace-output-dir 0 ../data/inputs/input2 &
python2 image_transformer.py --output-dir ../data/inputs/transformed/ --image-ext jpg --replace-output-dir 0 ../data/inputs/input3 &
python2 image_transformer.py --output-dir ../data/inputs/transformed/ --image-ext jpg --replace-output-dir 0 ../data/inputs/input4 &
python2 image_transformer.py --output-dir ../data/inputs/transformed/ --image-ext jpg --replace-output-dir 0 ../data/inputs/input5 &
python2 image_transformer.py --output-dir ../data/inputs/transformed/ --image-ext jpg --replace-output-dir 0 ../data/inputs/input6 &
python2 image_transformer.py --output-dir ../data/inputs/transformed/ --image-ext jpg --replace-output-dir 0 ../data/inputs/input7 &

echo "Waiting for all jobs to complete..."
wait
end=`date +%s`
runtime=$((end-start))
echo "image_transformer.py completed! Took $runtime seconds."






