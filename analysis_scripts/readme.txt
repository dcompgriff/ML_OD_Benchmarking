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