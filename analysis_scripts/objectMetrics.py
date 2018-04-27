
import os
import glob
import json
import pandas# as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pp


dataF = '../data/'
anntF = dataF+'inputs/annotations/instances_val2017.json'
cocoGt=COCO(anntF)



def objInfo(img,cocoGt):
    scores = []
    imgId = img['image_id']
    ##Handle detected objects
    matches = {'None':0}
    #Skip objects with low confidence, they are not classified at all
    objects = [_ for _ in img['objects'] if _['score']>=0.5 ]
    #find number of objects matching to the same annotation
    for obj in objects:
        id = obj['matchObjectId']
        id = 'None' if id is None else str(id)#include 'None' match as well
        matches[id] = (matches[id] if id in matches else 0)+1#increment count
    for obj in objects:
        #just need info on predicted and actual class.
        #actual class is class of bbox this object matched to.
        #actual class could be None, if this object didnt match to any bbox
        id = obj['matchObjectId']
        id = 'None' if id is None else str(id)
        scores.append({'predicted':obj['detClass'],
                       'actual':obj['matchClass'],
                       'matchId': id,#string here
                       'wt':(1/matches[id]) if id in matches else 0,
                       'score':obj['score'],
                       'image_id':imgId,
                       'transform':img['transform']})
        #TP = (predicted==class and actual==class)*wt
        #FP = (predicted==class and actual!=class)*wt
        #weighting done as many detected objects are matched to same annotated object. 
        #total wt per annotated object = 1, including 'None' match
    ##Handle undetected objects
    dt = pandas.DataFrame(cocoGt.loadAnns(cocoGt.getAnnIds(imgIds = [imgId])))
    for _,obj in dt.iterrows():
        id = str(obj['id'])
        if id not in matches:
            scores.append({'predicted':None,
                           'actual':obj['category_id'],
                           'matchId': id,#string here
                           'wt':1,#annotation that were not detected have Wt=1 for each as above
                           'score':0,
                           'image_id':imgId,
                           'transform':img['transform']})#0 as NN did not predict this object
        #FN = (actual==class and predicted!=class)*wt
        #TN = (actual!=class and predicted!=class)*wt
    
    return scores
#def imageInfo(img):
#    imgId = img['image_id']
#    det = len(pd.DataFrame(cocoGt.loadAnns(cocoGt.getAnnIds(imgIds = [imgId]))))
#    FN =  det- img['uniqueMatches']
#    imginfo = {
#        'id':img['image_id'],
#        'transform':img['transform'],
#        'total_detected':img['totalDetected'],
#        'total_annotated':det,#total distinct objects annotated
#        'matched':img['matched'],
#        'unique_matches':img['uniqueMatches'],#total unique objects detected
#        }#annotation that were not detected. Wt=1 for each similar to objInfo
#    return imginfo
#folder that contains json files with matches for a single model and a single transform
def modelObjectScores(jsons,cocoGt):
    objects= []
    for i, js in enumerate(jsons):
        #if i>0:
        #    break
        #print(js)
        with open(js) as fd:
            imgs = json.load(fd)
        for _,img in imgs.items():
            objects = objects+objInfo(img,cocoGt)
    return objects



valueTransforms = [
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
 'elastic_1']
spaceTransforms = [
 'scaled_1p25',
 'scaled_0p75',
 'scaled_0p5',
 'scaled_(1p25, 1p0)',
 'scaled_(0p75, 1p0)',
 'scaled_(1p0, 1p25)',
 'scaled_(1p0, 0p75)',
 'translate_(0p1, 0p1)',
 'translate_(0p1, -0p1)',
 'translate_(-0p1, 0p1)',
 'translate_(-0p1, -0p1)',
 'translate_(0p1, 0)',
 'translate_(-0p1, 0)',
 'translate_(0, 0p1)',
 'translate_(0, -0p1)',
 'rotated_3',
 'rotated_5',
 'rotated_10',
 'rotated_45',
 'rotated_60',
 'rotated_90',
 'flipH',
 'flipV',
 'dropout']
print('total value transforms: ',len(valueTransforms), 'totla space transforms: ',len(spaceTransforms))

#Files to read
models = ['e2e_faster_rcnn_R-50-C4_2x',
'e2e_faster_rcnn_R-50-FPN_2x',
'e2e_faster_rcnn_R-101-FPN_2x',
'e2e_faster_rcnn_X-101-64x4d-FPN_2x',
'e2e_mask_rcnn_R-50-C4_2x',
'e2e_mask_rcnn_R-50-FPN_2x',
'e2e_mask_rcnn_R-101-FPN_2x',
'retinanet_R-50-FPN_2x',
'retinanet_R-101-FPN_2x',
'retinanet_X-101-64x4d-FPN_2x']
outputs = dataF+'outputs/'

newResF = outputs +'5000_original_results_matches/'
newTranF = outputs + 'transformed_outputs_matches/'
newSpaceF = outputs + 'transformed_spatial_output_matches/'
#create new directories
out = outputs+'metrics/'
if not os.path.exists(out):
    os.makedirs(out)
#List of all transforms
pathsV = [[newTranF+_+'/',_] for _ in valueTransforms]
pathsS = [[newSpaceF+_+'/',_] for _ in spaceTransforms]
#objects = []
def toCSV(jsons, out, trsfm, algo,cocoGt):
    objects = modelObjectScores(jsons,cocoGt)
    dt = pandas.DataFrame(objects)
    #write scores to individial csv for each model,transformation
    dt.to_csv(out+trsfm+'_'+algo+'.csv', index=False)


job_server = pp.Server() 
jobs=[]
# Define your jobs
#job1 = job_server.submit(parallel_function, ("foo",))
#job2 = job_server.submit(parallel_function, ("bar",))

# Compute and retrieve answers for the jobs.
#print job1()
#print job2()


for algo in models:
    for path in [[newResF,'None']]+pathsV+pathsS:
        print('Model: ',algo, 'Transform: ', path[1])
        jsonF = path[0]+algo+'/'
        jsons = glob.iglob(jsonF + '*.json')
        #get scores of all objects in this
        jobs.append(job_server.submit(
            toCSV, 
            (list(jsons),out,path[1],algo,cocoGt), 
            (objInfo,modelObjectScores,), 
            ("json","pandas")))
print('x')

for job in jobs:
    job()