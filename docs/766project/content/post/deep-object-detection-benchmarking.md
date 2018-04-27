+++
date = "2018-04-23T13:50:46+02:00"
tags = []
title = "Deep object detection network benchmarking and sensitivity analysis."
+++

## Introduction

Deep object detection methods represent the pinnacle of cutting edge object detection methods.


## Background

##### Object Detection 

Dealing with object in images can loosely be broken down into 4 main categories (at the time of making this post anyways).
The first method of dealing with objects in images is "Object Classification". In this task, an entire image is taken to 
represent a single kind of object, and the goal is to classify the image. The next, more advanced task is Object Classification 
with "Localization". In this task, we try to determine two things. The first is where the object is within an image, and the second 
is to classigy what object is in that location. The next task is "Object Detection". This builds on Object Classification 
with Localization by finding multiple regions likely to have an object, then classifying each region as an individual object. The 
final task is "Instance Segmentation" (Image Segmentation), where the meaningful objects within an image are organized into 
singular segments. For our project, we focus on the "Object Detection" task. A visual representation of the different kinds of 
tasks is provided below.

**Object Detection Tasks**
![Object Detection Kinds](/img/obj_det_kinds.png "Object Detection Tasks")

As stated above, the Object Detection task is comprised of two parts. The first part deals with generating what are called 
"Object Proposals". Object Proposals are regions of an image that are considered likely to contain an object. The actual 
object proposal task can be performed by many algorithms. These object proposals are then passed to an Object 
Classification algorithms that classifies each region as an object (typically with some sort of confidence). Object Proposal 
and Object Classification used to be considered as two seprate tasks, requiring a dedicated algorithm for each. However, 
more recent Object Detection methods incorporate both the Object Proposal and Object Detection tasks into a single 
deep neural network architecture, thus creating a single network that can perform the full end to end Object Detection.

**Object Proposals and Classification**
![Object Proposals and Classification](/img/obj_proposals.png "Object Proposals and Classification")

##### Deep Object Detection Network Sensitivity

There has been increasing interest in deep neural network based Object Detection method's sensitivity to errors or 
variations in images. Currently, most of the focus is in the context of "Adverserial Attacks", in which an image is specifically 
designed to trick a deep neural network into misclassifying the image with high confidence. In this scenario, some deep 
neural network repeatedly probes an Object Detection network with an image, iteratively modifying it to a small degree 
until the deep neural network misclassifies it. In some cases, the modifications that need to be made to an image 
to trick a deep neural network are so small that they are not visible to the human eye. This issue is visually depicted 
below.

**Adverserial Object Detection**
![Adverserial Object Detection](/img/obj_detection_adverserial.png "Adverserial Object Detection")

Another kind of deep neural network sensitivity is a deep neural network's ability to correctly detect and classify 
objects in the presence of noise, or what we call "natural" image variation. We make this particular distinction 
from the adverserial case because these kinds of image variations are not mathematically constructed for 
the purpose of tricking a deep neural network. These are images that may have saturation, contrast, lighting, 
noise, blur, and other kinds of issues that may be naturally occuring. From our searches, we have not  been 
able to find anyone that has previously proposed this kind of distinction for image variation. In our project, we 
focus on these types of "natural" image variations, and analyse the sensitivity of deep neural networks to this 
kind of image variation. A visual representation of some of examples of natural image variations are given 
below.

**Natural Image Variation Object Detection**
![Natural Image Variation Object Detection](/img/obj_detection_natural_variation.png "Natural Image Variation Object Detection")


## Project Proposal

##### Problem?

For this project, we focused on 3 main questions related to deep neural network based object detection methods.

1. How sensitive are cutting edge deep object detection networks to natural noise/errors/variation in images?

2. What kind of process and metrics should be used to quantify this sensitivity?

3. Can our analysis be used to provide a context of which networks to use in different contexts?


##### Importance?

1. **Safety Applications**: A practical understanding or methodology for determining how Deepnets fail is crucial if we expect to use deep nets in safety concerned applications, such as autonomous driving.

2. **Context Specific Applications**: Field focus is on “new” networks.
Little knowledge of practical use “in the wild”.

3. **Lack of Rich Performance Metrics**: Standard ML metrics don’t map very well to object detection.
There aren’t rich metrics that quantify things like “color sensitivity”


##### State of the art?

We aim to benchmark some of the most commonly used, modern deep neural networks
including Mask R-CNN , RetinaNet , Faster R-CNN , RPN , Fast R-CNN , and R-FCN . Some of the
pre-trained backbone networks we are looking to use include ResNeXt{50,101,152} ,
ResNet{50,101,152} , Feature Pyramid Networks , and VGG16 . The datasets we aim to use
include the COCO dataset, PASCAL VOC dataset, and the Kitti object detection dataset. These
network architecture designs, and network backbones represent some of the most common
cutting edge object detection deep neural networks that exist today. 

In terms of actual analysis, not much work has been performed into analysis of deep
neural network models other than basic average performance metrics of individual models. Most
work focuses more on proposing a new architecture for a deep network object detection model,
rather than comparing its significant difference with existing methods, major fundamental
drawbacks, and other ways in which the network can be fooled or broken. We provide a general 
summary of some of the state of the art networks, backbones, data sets, and kinds of analysis.

+ Networks:
  + Mask R-CNN, RetinaNet, Faster R-CNN , RPN , Fast R-CNN , R-FCN, YOLO 
+ Backbones: 
  + ResNeXt{50,101,152} , ResNet{50,101,152} , Feature Pyramid Networks , and VGG16
+ Datasets:
  + COCO (Multi-object Detection Dataset)
  + PASCAL VOC (Multi-object Detection Dataset))
  + Kitti (Autonomous Vehicle Object Detection Dataset)
+ Analysis:
  + Simple aggregated performance metrics
  + Adversarial based analysis



##### Existing Systems or New Approach?

We plan on benchmarking existing neural network designs and implementations that are
representative of current deep neural network object detection based methods, with one
particular goal being a proposal and outline of future research work that should be developed to
address some of the issues we find. This might fall under the category of a “new approach”.
However, we also plan on performing deeper analysis of existing methods than has been
previously performed to provide for a better overall understanding of how well modern methods
compare, how similar and different the methods are, which approaches seem to work the best,
and what shortcoming exist with existing methods.

##### Evaluation and Results Plan?

Our project proposal centers around benchmarking and analyzing different deep object
detection networks, so the evaluation phase is really the meat of our project (aside from getting
everything up and running). The details of what we plan on benchmarking, what platforms we
plan to use, what data sets we plan to use, and what metrics we plan to calculate are given in
the “Approach” section below.

## Approach


##### High Level Steps

1. Select the most cutting edge object detection networks.
2. Perform performance analysis on a base set of images for each network.
3. Generate images with “natural” errors/noise/artifacts/transforms.
4. Perform performance analysis on transformed images for each network.
5. Compile sets of summary metrics organized by object classes, transform kind, and model.
6. Compare and contrast performance to draw insights about deep network capabilities and 
sensitivities based on class, transform kind, and model.
7. Create guidelines for developing and using deep object detection networks, and propose new data.

##### Networks, Images, Datasets

Below, we outline the networks we tested, the analysis we performed, and the systems we used.

+ 10 Networks (Requires 11GB Graphics Cards): 
  + \<Net Arch\>_\<Net Backbone\>-\<Proposals\>_2x
  + e2e_faster_rcnn_R-101-FPN_2x
  + e2e_faster_rcnn_R-50-C4_2x
  + e2e_faster_rcnn_R-50-FPN_2x
  + e2e_faster_rcnn_X-101-64x4d-FPN_2x
  + e2e_mask_rcnn_R-101-FPN_2x
  + e2e_mask_rcnn_R-50-C4_2x
  + e2e_mask_rcnn_R-50-FPN_2x
  + retinanet_R-101-FPN_2x
  + retinanet_R-50-FPN_2x
  + retinanet_X-101-64x4d-FPN_2x

+ Images:
  + COCO image validation data set. 
  + 5000 base images (600x800 3 channel images)
  + 50 Transforms generating 250,000 images total
  + 5000*50*10*(300 proposals/net) = 750 Million object candidates

+ Systems:
  + 8 GTX 1080ti GPU system with 11GB per Card, 8 GB Ram
  + i7 CPU, 24GB Ram
  + University of Wisconsin Condor GPU Instances

##### Image Transforms

We have 50 image transforms that we applied to the COCO validation image set. There 
are roughly two categories of image transforms that we applied. The first is value based 
transforms. Most of these transforms deal with changing images in ways related to the 
value of pixels in an image. The second is spatial based transforms, in which we transform 
images in ways that may alter the phisicaly size, shape, or arraingment of pixels within an 
image. We provide some visual examples of each kind of image transform below.


**Value Based Transforms**
![Value Based Transforms](/img/value_transforms.png "Value Based Transforms")

**Spatially Based Transforms**
![Spatially Based Transforms](/img/spatial_transforms.png "Spatially Based Transforms")

## System

##### Systems Overview

![Pipeline](/img/pipeline.png "Pipeline")

##### Languages and Frameworks Utilized
S


## Results

<table class="table">
    <tr>
        <th>Name</th>
        <th>Age</th>
    </tr>
    <tr>
        <td>Dan</td>
        <td>21</td>
    </tr>
    <tr>
        <td>Jane</td>
        <td>45</td>
    </tr>
</table>


## Discussion

##### Short Summary

![Short Summary](/img/short_summary_meme.png)



## Future Work

##### Metrics Specific to Error/Variation Sensitivity for Object Detection

Modern machine learning performance metrics are generally centered on precision, recall, AUC for ROC curves, 
Precision/Recall Curves, and 

## Special Thanks

Special thanks to Matt Nichols <a href="//github.com/mattuyw" class="icon-github" target="_blank" title="Github"></a> for providing us access to his 8 GPU bitcoin mining machine! 
Without access to such a powerful system, this project would not have been possible. 


## Appendix

##### Challenges We Faced

Refactoring our project and proposal.

+ So we had to rapidly change our direction when we realized the tedious
complexity of our original project proposal, and search for a new potential project.
We had to review lots of object detection papers and methods to come up with a
new project to work on. Once we had an idea that we wanted to try to perform a
set of benchmarking, we had to find candidate models, platforms, frameworks,
datasets, and metrics that we thought we could achieveably use for our project.
Some of the challenges of these are found in the next subsections.

Running on Condor GPUs and getting access to resources.

+ Most of today’s existing deep neural network, object detection based models
require GPUs (A major drawback which I believe will eventually be remedied with
FPGAs due to their high speed, low power nature which makes them a defacto
choice for most industry scale, long term use). So, we talked to the University of
Wisconsin’s HPC group about running on the GPU based condor instances. We
spent a good amount of time writing the submission and monitoring scripts for the
GPU instances, and in particular the docker based GPU instances to simplify
code submission. However, there are only 2 machines in HPC with 4 total GPUs
that currently run docker, and the queue is so congested that using condor for
our project became clearly impractical (I submitted a GPU based docker job
about a week ago that still hasn’t been run yet). I’ve talked with the condor HPC
Team Members - Daniel Griffin, Yudhister Satija
staff, and they are working on getting docker running on the other 2 older
instances (each with 6-8 GPUs each) that don’t have Docker (But we can’t use
these without docker, as the nature of the code we are trying to run on the
machines requires installation and system configuration permissions).

Running on AWS, and the cost associated with it.

+ Since we couldn’t run on the condor instances, we decided to turn to Amazon
Web Services (AWS) to try to provision instances. After a few days of working
with AWS EC2 instances, we finally developed a process for provisioning AWS
GPU instance resources for running our nvidia-docker based code, and running
some of our initial deep net object detection models (Many of which need GPUs
for their operators, and up to 15GB of dedicated system memory). The downside
with this is that AWS GPU instances are $1/HR to provision and use which is
quite expensive for compute resources (And why we are trying to avoid training
models that can each take anywhere from hours to days to train).

Dataset non-uniformity.

+ While there exist some different datasets for benchmarking object detection,
most data sets have different formats. We have decided to try to standardize to
the COCO dataset API, and to try to convert some of the other data sets we
would like to use to the COCO annotation specifications. This means that we will
likely have to write scripts for generating COCO annotated images from each
data set.


### Initial Proposal and Midterm Reports.

+ [Initial Project Proposal](/publications/766_CV_Project_Proposal.pdf)
+ [Midterm Report](/publications/766_CV_Midterm_Report.pdf)
+ [Project Presentation](/publications/766_Final_Project_Presentation.pptx)




