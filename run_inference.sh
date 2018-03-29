#!/bin/bash

# Run the demo CNN image labeling.
mkdir /tmp/test_out
python2 tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir /tmp/test_out/ \
    --image-ext jpg \
    --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    demo

# Zip the output files in output.zip
#zip -r out.zip /tmp/test_out/
tar -cvf output.tar /tmp/test_out



