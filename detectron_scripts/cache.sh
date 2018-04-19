#!/bin/bash

echo "Downloading all object detection models..."
echo "NOTE: This script assumes that it was ran from the root github directory."

# Generate start time.
start=`date +%s`

#############################################################################################
#Faster RCNN Networks.
#############################################################################################

# Run code for R-50-C4-2x outputs.
mkdir ./detectron-download-cache/e2e_faster_rcnn_R-50-C4_2x.yaml
wget -P ./detectron-download-cache/e2e_faster_rcnn_R-50-C4_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl

# Run code for R-50-RPN-2x
mkdir ./detectron-download-cache/e2e_faster_rcnn_R-50-FPN_2x.yaml
wget -P ./detectron-download-cache/e2e_faster_rcnn_R-50-FPN_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/35857389/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml.01_37_22.KSeq0b5q/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl

# Run code for R-101-FPN-2x
mkdir ./detectron-download-cache/e2e_faster_rcnn_R-101-FPN_2x.yaml
wget -P ./detectron-download-cache/e2e_faster_rcnn_R-101-FPN_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/35857952/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml.01_39_49.JPwJDh92/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl

# Run code for X-101-64x4d-FPN-2x
mkdir ./detectron-download-cache/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml
wget -P ./detectron-download-cache/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/35858198/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml.01_41_46.CX2InaoG/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl

# Run code for MaskRCNN R-50-RPN-2x 
mkdir ./detectron-download-cache/e2e_mask_rcnn_R-50-C4_2x.yaml
wget -P ./detectron-download-cache/e2e_mask_rcnn_R-50-C4_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/35858828/12_2017_baselines/e2e_mask_rcnn_R-50-C4_2x.yaml.01_46_47.HBThTerB/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl

# Run code for MaskRCNN R-50-FPN-2x
mkdir ./detectron-download-cache/e2e_mask_rcnn_R-50-FPN_2x.yaml
wget -P ./detectron-download-cache/e2e_mask_rcnn_R-50-FPN_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl

# Run code for MaskRCNN R-101-FPN-2x
mkdir ./detectron-download-cache/e2e_mask_rcnn_R-101-FPN_2x.yaml
wget -P ./detectron-download-cache/e2e_mask_rcnn_R-101-FPN_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl

#(NOTE: This currently errors out!) Run code for MaskRCNN X-101-64X4d-FPN-2x
#mkdir ./detectron-download-cache/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml
#wget -P ./detectron-download-cache/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/35859745/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml.02_00_30.ESWbND2w/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl

# Run code for Retinanet R-50-FPN-2X
mkdir ./detectron-download-cache/retinanet_R-50-FPN_2x.yaml
wget -P ./detectron-download-cache/retinanet_R-50-FPN_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/36768677/12_2017_baselines/retinanet_R-50-FPN_2x.yaml.08_30_38.sgZIQZQ5/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl

# Run code for Retinanet R-101-FPN-2x
mkdir ./detectron-download-cache/retinanet_R-101-FPN_2x.yaml
wget -P ./detectron-download-cache/retinanet_R-101-FPN_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/36768840/12_2017_baselines/retinanet_R-101-FPN_2x.yaml.08_33_29.grtM0RTf/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl

# Run code for Retinanet X-101-64X4d-FPN-2x
mkdir ./detectron-download-cache/retinanet_X-101-64x4d-FPN_2x.yaml
wget -P ./detectron-download-cache/retinanet_X-101-64x4d-FPN_2x.yaml https://s3-us-west-2.amazonaws.com/detectron/36768907/12_2017_baselines/retinanet_X-101-64x4d-FPN_2x.yaml.08_35_40.pF3nzPpu/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl

echo "Downloaded all .pkl nets!"
end=`date +%s`
runtime=$((end-start))
echo "Total runtime was $runtime seconds."






