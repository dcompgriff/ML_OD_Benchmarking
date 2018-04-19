#!/bin/bash

echo "NOTE: To run the reduced version of this script, run './benchmark test'"
# Get argument for running the benchmark script.
ARG1=${1:-full}

if [ $ARG1 == "test" ]
then
# "test" argument was given.
# Run code for only first network.
echo "Running first object detection model only..."    

# Run code for R-50-C4 2x outputs.
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

elif [ $ARG1 == "full" ]
then 
# No "test" argument was given.
# Run code for other models...
echo "Running all object detection models..."

# Generate start time.
start=`date +%s`

#############################################################################################
#Faster RCNN Networks.
#############################################################################################

# Run code for R-50-C4-2x outputs.
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	    /ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

# Run code for R-50-RPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35857389/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml.01_37_22.KSeq0b5q/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

# Run code for R-101-FPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts  https://s3-us-west-2.amazonaws.com/detectron/35857952/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml.01_39_49.JPwJDh92/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	        /ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

# Run code for X-101-64x4d-FPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts  https://s3-us-west-2.amazonaws.com/detectron/35858198/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml.01_41_46.CX2InaoG/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

# Run code for X-101-32X8D-FPN-2x
#python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
#	--cfg /detectron/configs/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_2x.yaml \
#	--output-dir /ML_OD_Benchmarking/data/outputs \
#	--image-ext jpg \
#	--wts  https://s3-us-west-2.amazonaws.com/detectron/36761786/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_2x.yaml.06_33_22.VqFNuxk6/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
#	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

#############################################################################################
#MaskRCNN Networks.
#############################################################################################

# Run code for MaskRCNN R-50-RPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-50-C4_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35858828/12_2017_baselines/e2e_mask_rcnn_R-50-C4_2x.yaml.01_46_47.HBThTerB/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

# Run code for MaskRCNN R-50-FPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

# Run code for MaskRCNN R-101-FPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs

rm -r /tmp/detectron-download-cache

# Run code for MaskRCNN X-101-64X4d-FPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35859745/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml.02_00_30.ESWbND2w/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	/ML_OD_Benchmarking/data/input

#rm -r /tmp/detectron-download-cache

# Run code for MaskRCNN X-101-32X8d-FPN-2x
#python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
#	--cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_2x.yaml \
#	--output-dir /ML_OD_Benchmarking/data/outputs \
#	--image-ext jpg \
#	--wts https://s3-us-west-2.amazonaws.com/detectron/36762092/12_2017_baselines/e2e_mask_rcnn_X-101-32x8d-FPN_2x.yaml.06_37_59.DM5gJYRF/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
#	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

#############################################################################################
#Retina net.
#############################################################################################

# Run code for Retinanet R-50-FPN-2X
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/retinanet_R-50-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/36768677/12_2017_baselines/retinanet_R-50-FPN_2x.yaml.08_30_38.sgZIQZQ5/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

# Run code for Retinanet R-101-FPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/retinanet_R-101-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https:/p/s3-us-west-2.amazonaws.com/detectron/36768840/12_2017_baselines/retinanet_R-101-FPN_2x.yaml.08_33_29.grtM0RTf/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

# Run code for Retinanet X-101-64X4d-FPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
	--cfg /detectron/configs/12_2017_baselines/retinanet_X-101-64x4d-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/36768907/12_2017_baselines/retinanet_X-101-64x4d-FPN_2x.yaml.08_35_40.pF3nzPpu/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

# Run code for Retinanet X-101-32X8d-FPN-2x
#python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
#	--cfg /detectron/configs/12_2017_baselines/retinanet_X-101-32x8d-FPN_2x.yaml \
#	--output-dir /ML_OD_Benchmarking/data/outputs \
#	--image-ext jpg \
#	--wts https://s3-us-west-2.amazonaws.com/detectron/36769641/12_2017_baselines/retinanet_X-101-32x8d-FPN_2x.yaml.08_42_55.sUPnwXI5/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl \
#	/ML_OD_Benchmarking/data/inputs

#rm -r /tmp/detectron-download-cache

#############################################################################################
#The big boy network.
#############################################################################################

# Run code for THE BIG BOY. This network is massive, and takes 12+ seconds per image to detect.
#python2 /ML_OD_Benchmarking/detectron_scripts/cache_models.py \
#	--cfg /detectron/configs/12_2017_baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml \
#	--output-dir /ML_OD_Benchmarking/data/outputs \
#	--image-ext jpg \
#	--wts https://s3-us-west-2.amazonaws.com/detectron/37129812/12_2017_baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml.09_35_36.8pzTQKYK/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
#	/ML_OD_Benchmarking/data/inputs &

#rm -r /tmp/detectron-download-cache

echo "Generated all jobs, waiting for jobs to complete..."
end=`date +%s`
runtime=$((end-start))
echo "All jobs completed! Total runtime was $runtime seconds."

fi






