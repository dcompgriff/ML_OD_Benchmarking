#Uses Python 3

import os
import glob
import json
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



import imgaug as ia
from imgaug import augmenters as iaa


#Names of value transforms handled. 
valueTransforms = ['gaussianblur_1',
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
 'elastic_1',
 'invert']#Unused
#names listed, though not used anywhere in script
spaceTransforms2 = ['scaled_1p25',
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
 'flipV'
 #'piecewiseAffine_0p01',#Removed as was getting error with bbox transformations
 #'piecewiseAffine_0p03',#
 #'piecewiseAffine_0p06',#
 #'piecewiseAffine_0p1'#
 ]
len(spaceTransforms2)


#Actually build the space transforms as a dictionary so as to use the transforms on bboxes

##Space transforms
spaceTransforms = {}
# Change image scale keeping aspect ratio same ( same scaling of width and height)
for r in [1.25,.75,.5]:
    spaceTransforms['scaled_' + str(r).replace('.','p')] = iaa.Sequential([
        iaa.Affine(scale=r)
    ])
# Change image scale not keeping aspect ratio same ( unequal scaling of width and height)
for r in [(1.25,1.0), (.75,1.0), (1.0,1.25), (1.0,.75)]:#125% or 75% of one of the dimensions
    spaceTransforms['scaled_' + str(r).replace('.','p')]=iaa.Sequential([
        iaa.Affine(scale={"x": r[0], "y": r[1]})
    ])
    
# translate image by +-10% towards each of corners(diagonally) or edges(horizontally and vertically)
for tr in [(.1,.1), (.1,-.1), (-.1,.1), (-.1,-.1), (.1,0), (-.1,0), (0,.1), (0,-.1)]:
    spaceTransforms['translate_' + str(tr).replace('.','p')]=iaa.Sequential([
        iaa.Affine(translate_percent={"x":tr[0], "y":tr[1]})
    ])
    
# Change orientation by few small angles and few large angles
for theta in [3,5,10,45,60,90]:
    spaceTransforms['rotated_' + str(theta).replace('.','p')]=iaa.Sequential([
        iaa.Affine(rotate=theta)
    ])
    
# Flip images horizontly
for _ in [1]:
    spaceTransforms['flipH']=iaa.Sequential([
        iaa.Fliplr(1.0)
    ])
    
# Flip images vertically
for _ in [1]:
    spaceTransforms['flipV']=iaa.Sequential([
        iaa.Flipud(1.0)
    ])
# Random pixel dropout, 10% of pixels, like salt and pepper noise without salt
for _ in [1]:
    spaceTransforms['dropout'] = iaa.Sequential([
        iaa.Dropout(p=0.1)
    ])   
# Not using this as raised error with bbox transformations
#for z in [0.01,0.03,0.06,0.1]:
#    spaceTransforms['piecewiseAffine_'+str(z).replace('.','p')]=iaa.Sequential([
#        iaa.PiecewiseAffine(scale=z)
#    ])
print("Total space transforms used: ", len(spaceTransforms))
print("Total value transforms used: ", len(valueTransforms))



#Files to read, hardcoded strings
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
print("Total models to test over: ", len(models))
#relative path
dataF = '../data/'
outputs = dataF+'outputs/'
#NN outputs
resF = outputs +'5000_original_results/'
tranF = outputs + 'transformed_outputs/'
spaceF = outputs + 'transformed_spatial_output/'
#Outputs with matches after running this script
newResF = outputs +'5000_original_results_matches/'
newTranF = outputs + 'transformed_outputs_matches/'
newSpaceF = outputs + 'transformed_spatial_output_matches/'
#create new directories to save this scripts outputs
pathsV = [newTranF+_+'/' for _ in valueTransforms]
pathsS = [newSpaceF+_+'/' for _ in spaceTransforms]
for path in [newResF]+pathsV+pathsS:
    for algo in models:
        directory = os.path.dirname(path+algo+'/')
        if not os.path.exists(directory):
            os.makedirs(directory)
#COCO annotations file
anntF = dataF+'inputs/annotations/instances_val2017.json'
#Load annotations
cocoGt=COCO(anntF)

#imgIds = cocoGt.getImgIds()
#annIds = cocoGt.getAnnIds(imgIds = [530162])
#anns = cocoGt.loadAnns(annIds)
#dt = pd.DataFrame(anns)
#dt


#Define a matching score, between two bounding boxes using intersection area over base area
#Used to find which nnotated bbox this bbox is most similar to.
#Not a good measure, but the best search method for now
def bbMatch(A,B, base = 'union'):
    '''Returns a Jaccard like score of match between the two bounding boxes. IntersectionArea / BaseArea'''
    areaA = A[2]*A[3]#First object's area
    areaB = B[2]*B[3]#Second objects area
    #print(areaU)
    if areaA==0 or areaB ==0:
        return 0#avoid division by zero when no match
    xa1,ya1,xa2,ya2 = A[0],A[1],A[0]+A[2], A[1]+A[3]
    xb1,yb1,xb2,yb2 = B[0],B[1],B[0]+B[2], B[1]+B[3]
    dx = min(xa2, xb2) - max(xa1, xb1)#overlap in x
    dy = min(ya2, yb2) - max(ya1, yb1)#overlap in y
    areaI= dx*dy if (dx>=0) and (dy>=0) else 0
    #Possible base areas
    area = {'first':   areaA,
            'second':  areaB,
            'sum':     areaA+areaB,
            'union':   areaA+areaB-areaI,#union = sum - intersection
            'larger':  areaA if areaA>areaB else areaB,
            'smaller': areaA if areaA<areaB else areaB}
    
    return areaI/area[base];

#A = [415.11, 204.57, 172.01, 258.54]
#B = [423.078522, 205.677017, 580.738586, 441.871582]
#bbMatch(A,B)

#Main part of the script. Runs over a group of jsons inside a NN model's output folder

def findMatchingBBoxes(detectedJsons, cocoAnnotations, outDir, algo):
    '''loopes over all images in Jsons generated by a single NN model.
    For each object in each of these images, finds the best matching object
    from among the coco Annotations for a particular image.
    Match is done by comparing overlapping area (Jaccard measure) of the two bounding boxes.
    Must match atleast 10%. Also only looks for objects labeled with confidence of atleast 50%
    Also creates a new json file that is more structured'''
    #loop over a whole folder full of output jsons
    #have 12 jsons for each model for value transforms, 14 jsons/model for space transforms, 1667 jsons/model for direct output 
    for i, js in enumerate(detectedJsons):#multiple json files returned by glob
        print(os.path.basename(js))#for bookeeping
        with open(js) as fd:
            imgs = json.load(fd)#about 3000 images per json for transformed cases, 3 images per json for direct output case
        outputs = []#initialize empty array. Helps in indexing
        for _idx,img in enumerate(imgs):#for each image mentioned in the json
            img_name = img['img_name'].split('/')[-1]#truncate to get just the file name
            parts = img_name.split('__')#only if transformed
            if len(parts)==2:#extract the transformation name
                transform = parts[0]
                imgId = int(parts[1][:-4])
                if 'piecewise' in transform:#skip all piecewise
                    continue
            else:
                imgId = int(parts[0][:-4])
                transform = None
            detected = []#contains one output per detected object
            count = 0#total matched
            uniques = set()#unique matches
            dt = pd.DataFrame(cocoAnnotations.loadAnns(cocoGt.getAnnIds(imgIds = [imgId])))#load coco data for this image
            for index,A in enumerate(img['bboxes']):#for each object detected by NN
                #reframe information to write in json
                info ={'detClass':img['classes'][index],
                        'detBbox':A,
                        'score':img['scores'][index],
                        'matchObjectId':None,
                        'matchClass':None,
                        'matchBBox':None,}
                detected.append(info)#Add the detected object
                if img['scores'][index]<0.5:#skip matching for objects with low confidence
                    continue
                best, bestM = None,-1#initial best guess
                for _, obj in dt.iterrows():#go over all given annotations for this image
                    B = obj.bbox#we use only bbox to match
                    if transform in spaceTransforms:#need to transform bbox
                        bbs = ia.BoundingBoxesOnImage(#make from x,y,w,h to x1,y1,x2,y2 for imgaug format
                            [ia.BoundingBox(x1=B[0], y1=B[1], x2=B[0]+B[2], y2=B[1]+B[3])],
                            shape=[cocoGt.imgs[imgId]['height'], cocoGt.imgs[imgId]['width']])#need image properties
                        T = spaceTransforms[transform]#iaa object 
                        try:#some transforms break, hence in try block
                            B = T.augment_bounding_boxes([bbs])[0].bounding_boxes[0]
                            B = [B.x1, B.y1, B.x2-B.x1, B.y2-B.y1]#reframe B to x,y,w,h
                        except:#failed transform
                            #print(B,transform)
                            continue#skip matching against this annotation
                            
                    #Use larger as base of match, for better result as prevents saturation (match=1)
                    m = bbMatch(A,B, 'larger')#find match between two objects' bboxes
                    if m >=0.10 and m>bestM:#best of all matches with atleast 10% match
                        best,bestM = obj,m
                if not best is None:#found some match, update properties
                    info['matchObjectId']=best.id
                    info['matchClass']=best.category_id
                    info['matchBBox']=best.bbox
                    uniques.add(best.id)
                    count = count+1#update count of matches
            #Save matchings information for this image into its correct index
            outputs.append({'image_id':imgId,
                             'objects':detected,#all objects detected in this image
                            'totalDetected': len(img['bboxes']),
                            'matched':count,
                            'uniqueMatches':len(uniques),
                            'transform':transform})#which transform was used
        #After completing one json file
        #Divide results into separate transformations, and write to corresponding directories
        #Keep same name for json files
        dt = pd.DataFrame(outputs)
        trsfms = dt['transform'].unique()
        if len(trsfms) == 1 and trsfms[0] is None:#from orig 5000 results, not transformed
            path = outDir + algo+'/'+os.path.basename(js)
            with open(path,'w') as fd:
                fd.write(json.dumps(dt.to_dict('index')))
        else:#a transformed result
            for tr in trsfms:
                path = outDir + tr + '/' + algo+'/'+os.path.basename(js)
                with open(path,'w') as fd:
                    fd.write(json.dumps(dt[dt['transform']==tr].to_dict('index')))
    
for algo in models[2:3]:
    print("Running for model: "+algo)
    origJsons = glob.iglob(resF+algo + '/*.json')
    print("starting on original results")
    #findMatchingBBoxes(origJsons, cocoGt,newResF,algo)
    origJsons = glob.iglob(tranF+algo + '/*.json')
    print("starting on value transformed results")
    #findMatchingBBoxes(origJsons, cocoGt,newTranF,algo)
    origJsons = glob.iglob(spaceF+algo + '/*.json')
    print("starting on space transformed results")
    findMatchingBBoxes(origJsons, cocoGt,newSpaceF,algo)




