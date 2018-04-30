import argparse
import os
import glob
import json
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
import time

#all allowed options
transformNames = [
 'None',
 'gaussianblur_1',
 'gaussianblur_10',
 'gaussianblur_20',
 'superpixels_0p1',
 'superpixels_0p5',
 'superpixels_0p85',
 'colorspace_25',
 'colorspace_50',
 'averageblur_5_11',
 'medianblur_1',
 'sharpen_0',
 'sharpen_1',
 'sharpen_2',
 'addintensity_-80',
 'addintensity_80',
 'elementrandomintensity_1',
 'multiplyintensity_0p25',
 'multiplyintensity_2',
 'contrastnormalization_0',
 'contrastnormalization_1',
 'contrastnormalization_2',
 'elastic_1'
]
#  'scaled_1p25',
#  'scaled_0p75',
#  'scaled_0p5',
#  'scaled_(1p25, 1p0)',
#  'scaled_(0p75, 1p0)',
#  'scaled_(1p0, 1p25)',
#  'scaled_(1p0, 0p75)',
#  'translate_(0p1, 0p1)',
#  'translate_(0p1, -0p1)',
#  'translate_(-0p1, 0p1)',
#  'translate_(-0p1, -0p1)',
#  'translate_(0p1, 0)',
#  'translate_(-0p1, 0)',
#  'translate_(0, 0p1)',
#  'translate_(0, -0p1)',
#  'rotated_3',
#  'rotated_5',
#  'rotated_10',
#  'rotated_45',
#  'rotated_60',
#  'rotated_90',
#  'flipH',
#  'flipV',
#  'dropout',
#  ]
modelNames = ['e2e_faster_rcnn_R-50-C4_2x',
'e2e_faster_rcnn_R-50-FPN_2x',
'e2e_faster_rcnn_R-101-FPN_2x',
'e2e_faster_rcnn_X-101-64x4d-FPN_2x',
'e2e_mask_rcnn_R-50-C4_2x',
'e2e_mask_rcnn_R-50-FPN_2x',
'e2e_mask_rcnn_R-101-FPN_2x',
'retinanet_R-50-FPN_2x',
'retinanet_R-101-FPN_2x',
'retinanet_X-101-64x4d-FPN_2x']

# Build path to transformed image outputs.
dataF = '../data/'
outputs = dataF+'outputs/'
tranF = outputs + 'transformed_outputs/'
resF = outputs +'5000_original_results/'
# Build path to outputs from this script.
newTranF = outputs + 'transformed_outputs_coco/'
newResF = outputs +'5000_original_results_coco/'

# Create new directories to save this scripts outputs.
if not os.path.exists(newTranF):
    os.makedirs(newTranF)
if not os.path.exists(newResF):
    os.makedirs(newResF)
# Create new sub-directories for each model.
for basePath in [newTranF, newResF]:
    for algo in modelNames:
        directory = os.path.dirname(basePath+algo+'/')
        if not os.path.exists(directory):
            os.makedirs(directory)


def generateOutputFiles(origJsons, args, allTransforms=True):
    # Stores the unrolled json output object files.
    objectDict = {transform: [] for transform in transformNames}

    for i, js in enumerate(origJsons):#Loop over each json file.
        print(os.path.basename(js))#For bookeeping
        # Extract the model name from the file name.
        algo = '_'.join(os.path.basename(js).split('_')[1:]).split('.json')[0]
        with open(js) as fd:
            imgs = json.load(fd)#about 3000 images per json for transformed cases, 3 images per json for direct output case

        for _idx,img in enumerate(imgs):#Loop over each image mentioned in the json
            img_name = img['img_name'].split('/')[-1]#truncate to get just the file name
            parts = img_name.split('__')  # only if output is the transformed output
            if allTransforms:
                if len(parts) == 2:
                    transform = parts[0]
                    imgId = int(parts[1][:-4])
                else:
                    imgId = int(parts[0][:-4])
                    transform = 'None'
            else:
                imgId = int(parts[0][:-4])
                transform = 'None'

            # Unroll the image into a list of json objects.
            imgDetectionList = []
            for i in range(len(img['scores'])):
                newObject = {"image_id": imgId, "category_id": img['classes'][i],"bbox": img['bboxes'][i],"score": img['scores'][i]}

                if float(newObject['score']) >= float(args.threshold):
                    imgDetectionList.append(newObject)
                else:
                    continue
            # Put the unrolled image object list into the appropriate list in the object dictionary.
            objectDict[transform].extend(imgDetectionList)

    # Save the json ouputs per model, per transform in a file.
    if allTransforms:
        path = newTranF + algo + '/'
    else:
        path = newResF + algo + '/'
    for key in objectDict.keys():
        if len(objectDict[key]) != 0:
            with open(path + key + '.json', 'w') as f:
                f.write(json.dumps(objectDict[key]))


def main(args):
    # For each model, parse it's ouput into transform specific json lists.
    initialStartTime = time.time()
    for algo in modelNames:
        startTime = time.time()
        if args.t.lower() == 'none':
            print("Running for model: " + algo)
            origJsons = glob.iglob(resF + algo + '/*.json')
            print("Starting on original results...")
            generateOutputFiles(origJsons, args, allTransforms=False)
        else:
            print("Running for model: " + algo)
            origJsons = glob.iglob(tranF + algo + '/*.json')
            print("Starting on original results...")
            generateOutputFiles(origJsons, args)
        endTime = time.time()
        print("Done! Took %s seconds." % (str(endTime - startTime)))
    finalEndTime = time.time()
    print("Totally Done! Entire time took %s seconds." % (str(finalEndTime - initialStartTime)))







        
            

if __name__ == '__main__':
    #Parse command line arguments
    parser = argparse.ArgumentParser(description="""After having metrics folder under data/outputs, use this script to generate a variety of metrics.\n
    Specify which model to use, which transform to use (or 'all') and which class to use (either id or class/sueprclass name) (or 'all')""")
    parser.add_argument('-t', metavar='Class', type=str, default='All', help="""Specify 'None' if outputs are for non-transformed images.""")
    parser.add_argument('-threshold', metavar='Class', type=str, default='0.0',
                        help="""Specify the threshold for the score of each object.""")
    args = parser.parse_args()
    main(args)

