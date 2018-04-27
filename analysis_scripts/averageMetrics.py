import argparse
import pandas as pd
from pycocotools.coco import COCO
import numpy as np

dataF = '../data/'
metricF = dataF+'outputs/metrics/'
anntF = dataF+'inputs/annotations/instances_val2017.json'


#partition data on a single model and a single transform
def partition(model,transform,classes):
    file = metricF+transform+'_'+model+'.csv'
    dt = pd.read_csv(file)
    dt = dt[dt['actual'].isin(classes) | dt['predicted'].isin(classes)]#Take both in order to get TP,FN,FN
    return dt
#partition data on a single model and multiple transforms
def multPartition(model, transforms, classes):
    prt = []
    for tr in transforms:
        prt.append(partition(model, tr, classes))
    return pd.concat(prt, ignore_index=True)
#calculate numbers for a partition
def TpFpFn(dt,Class):
    TP = dt[(dt.actual==Class) & (dt.predicted==Class)]
    FP = dt[(dt.actual!=Class) & (dt.predicted==Class)]
    FN = dt[(dt.actual==Class) & (dt.predicted!=Class)]
    TN = dt[(dt.actual!=Class) & (dt.predicted!=Class)]#Zero if partition is used
    return {'tp':sum(TP.wt),'fp':sum(FP.wt),'fn':sum(FN.wt),'tn':sum(TN.wt)}

def main(args):
    coco=COCO(anntF)

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
     'dropout',
     ]
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
    cats = coco.loadCats(coco.getCatIds())

    if args.m not in modelNames:
        raise ValueError('Model: '+args.m+' not found')
    model = args.m#Single model only, not all
    print('model: ',model)
    
    if args.t=='all':
        transform = transformNames
    elif args.t not in transformNames:
        raise ValueError('Transform: '+args.t+' not found')
    else:
        transform = [args.t]
    print('transforms: ',transform)
    if args.r in ['basic','diff']:#need class value
        if args.c is None:
            raise ValueError('Need a Class value (-c) for the metric '+args.r)
        elif args.c =='all':
            raise NotImplementedError('Not yet sure how to measure metrics this for multi-class case')#classes = [cat['id'] for cat in cats]
        else:
            try:
                id = int(args.c)
            except:
                cat = [cat for cat in cats if (cat['name'] == args.c or cat['supercategory']==args.c)]
                if len(cat)==0:
                    raise ValueError('Category name: '+args.c+' not found')
                else:
                    id = cat[0]['id']
            if id<1 or id>80:
                raise ValueError('Category ID: '+str(id)+' not found')
            classes = [id]
        print('classes: ',classes)

        dt = multPartition(model, transform, classes)
        met = TpFpFn(dt,classes[0])#currently allowed only a single class

        metrics = {
            'precision':met['tp']/(met['tp']+met['fp']),
            'recall':met['tp']/(met['tp']+met['fn']),
            'accuracy':(met['tp']+met['tn'])/sum(met.values()),
            'TP':met['tp'],
            'FP':met['fp'],
            'FN':met['fn'],
            #'TN':met['tn']
        }
        if args.r=='basic':
            print(metrics)
        else:#when we need diff
            dt = multPartition(model, ['None'], classes)#original results
            met = TpFpFn(dt,classes[0])#currently allowed only a single class
            metrics2 =  {
                'precision':met['tp']/(met['tp']+met['fp']),
                'recall':met['tp']/(met['tp']+met['fn']),
                'accuracy':(met['tp']+met['tn'])/sum(met.values()),
                'TP':met['tp'],
                'FP':met['fp'],
                'FN':met['fn'],
                #'TN':met['tn']
            }
            metrics3 = {key:(metrics2[key]-metrics[key]) for key in metrics}
            print('Original - tranformed')
            print(metrics3)
    elif args.r =='matrix':
        labels=[cat['id'] for cat in cats]
        dt = multPartition(model,transform,labels)
        matrix = [[sum(dt[(dt.actual==actual) & (dt.predicted==predicted)].wt) for actual in labels] for predicted in labels]
        mat=pd.DataFrame(np.matrix(matrix))
        mat.columns = ['A_'+str(i) for i in labels]
        mat.index = ['P_'+str(i) for i in labels]
        mat.to_csv("./Confusion_Matrix.csv")
    else:
        raise ValueError("Metric: "+args.r+" not found")
        
            

if __name__ == '__main__':
    #Parse command line arguments
    parser = argparse.ArgumentParser(description="""After having metrics folder under data/outputs, use this script to generate a variety of metrics.\n
    Specify which model to use, which transform to use (or 'all') and which class to use (either id or class/sueprclass name) (or 'all')""")
    parser.add_argument('m', metavar='model', type=str, help="Model name")
    parser.add_argument('t', metavar='transform', type=str, help="""Transform name, 'all' to use all""")
    parser.add_argument('-c', metavar='Class', type=str, default=None, help="""Integer id or string class/superclass name""")
    parser.add_argument('-r', metavar="metric",type=str, default='basic', help="'basic', 'matrix' or 'diff' as the result to show")
    args = parser.parse_args()
    main(args)

