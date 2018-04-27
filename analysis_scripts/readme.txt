averageMetrics.py
    finds average metrics as requested in google doc.
    3 basic metric types:
        'basic':
            prints (Pr, RC, Accuracy and TP,FP,FN numbers) for specified model, transform and class
        'diff':
            prints difference w.r.t transform='None' for (Pr, RC, Accuracy and TP,FP,FN numbers) for specified model, transform and class
        'matrix':
            prints to file a confusion matrix for specified model and transform.
    I have mentioned in --help how to use script.
    Specifically, you can give a single transform name or 'all' to run on all transforms together
    Class value is necessary for 'basic' or 'diff' metric
    Class value can be the class's id or name or name of supercategory if wish to club thos together

prepareMatches.py:

Written in python 3 with corresponding dependencies of coco API and imgaug

This script takes the NN outputs from all json files we have and does matching with coco annotations using bboxes. For spatial transforms, the coco bboxes
are also transformed. Not including peicewise transforms as they were unable to transform bboxes.

Required directory structure:
Repo(ML_OD_.....)
    analysis_scripts
        this script
    data
        inputs
            annotations
                instances_val2017.json (coco annotations file)
        outputs
            5000_original_results
                For each model, a folder
                    1667 json files of original results for this model
            transformed_outputs
                For each model, a folder
                    the 0-11 json files of value transformed results for this model
            transformed_spatial_output
                For each model, a folder
                    the 0-14 json files of space transformed results for this model
                    
Parallelization:
    Currently all for loops are written sequentially. But except for the for loop at line 240, all others have independent iterations and can be paralleized. 
    You can look into it if you want to.

Book keeping
    I am providing the log of which model, which results folder, and which json file is being currently done. If you run into errors and have to restart the script,
    you can shunt out the already done files using for loops at line 286 or line 208
    
Last thought, It's not necessary for you to run this script unless you are parallelizing it. I can just give you the results. Don't spend good time on running or debugging this. Focus more on your part of the project.