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
python2 /ML_OD_Benchmarking/detectron_scripts/infer_simple.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	    /ML_OD_Benchmarking/data/inputs

elif [ $ARG1 == "full" ]
then 
# No "test" argument was given.
# Run code for other models...
echo "Running all object detection models..."

# Generate start time.
start=`date +%s`

# Run code for R-50-C4-2x outputs.
python2 /ML_OD_Benchmarking/detectron_scripts/infer_simple.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	    /ML_OD_Benchmarking/data/inputs &
# Run code for R-50-RPN-2x
python2 /ML_OD_Benchmarking/detectron_scripts/infer_simple.py \
	--cfg /detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml \
	--output-dir /ML_OD_Benchmarking/data/outputs \
	--image-ext jpg \
	--wts https://s3-us-west-2.amazonaws.com/detectron/35857389/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml.01_37_22.KSeq0b5q/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	/ML_OD_Benchmarking/data/inputs &

echo "Generated all jobs, waiting for jobs to complete..."
wait
end=`date +%s`
runtime=$((end-start))
echo "All jobs completed! Total runtime was $runtime seconds."

fi






