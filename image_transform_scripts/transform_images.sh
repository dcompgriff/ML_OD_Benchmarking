#!/bin/bash

echo "Starting sequential image transform process."
start=`date +%s`
python2 image_transformer.py --output-dir ../data/inputs/transformed/ --image-ext jpg ../data/inputs
end=`date +%s`
runtime=$((end-start))
echo "image_transformer.py completed! Took $runtime seconds."






