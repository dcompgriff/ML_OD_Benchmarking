+++
date = "2018-04-23T13:50:46+02:00"
tags = []
title = "Deep object detection network benchmarking and sensitivity analysis."
+++

## Introduction

Deep object detection methods represent the pinnacle of cutting edge object detection methods.


## Background

Dealing with object in images can loosely be broken down into 4 main categories (at the time of making this post anyways).
The first method of dealing with objects in images is "Object Classification". In this task, an entire image is taken to 
represent a single kind of object, and the goal is to classify the image. The next, more advanced task is Object Classification 
with "Localization". In this task, we try to determine two things. The first is where the object is within an image, and the second 
is to classigy what object is in that location. The next task is "Object Detection". This builds on Object Classification 
with Localization by finding multiple regions likely to have an object, then classifying each region as an individual object. The 
final task is "Instance Segmentation" (Image Segmentation), where the meaningful objects within an image are organized into 
singular segments. For our project, we focus on the "Object Detection" task. A visual representation of the different kinds of 
tasks is provided below.

![Object Detection Kinds](/img/obj_det_kinds.png "Object Detection Tasks")




## Project Proposal

##### Problem?

1. How sensitive are cutting edge deep object detection networks to natural noise/errors/variation in images?

2. What kind of process and metrics should be used to quantify this sensitivity?

3. Can our analysis be used to provide a context of which networks to use in different contexts?


##### Importance?

While some simple benchmarking has been previously performed, there exists a need to
thoroughly benchmark and explore today’s existing cutting edge object detection models to
understand how they differ, potential shortcomings, and how new models may be designed that
improve upon issues that exist with today’s models. While it is known that deep neural networks
produce some of the highest detection performance, it is not yet fully understood if there are
good rules or guidelines for designing modern detection networks, and the full capabilities or
drawbacks of existing cutting edge detection networks are only understood at a cursory level.
Deep analysis of today’s cutting edge object detection neural networks is necessary to
understand where improvements should be made for future research, which methods should be
applied for different scenarios, when current methods perform within their suggested
performance bounds, and the potentially unstated drawbacks of current methods.


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
drawbacks, and other ways in which the network can be fooled or broken. 

##### Existing Systems or New Approach?

We plan on benchmarking existing neural network designs and implementations that are
representative of current deep neural network object detection based methods, with one
particular goal being a proposal and outline of future research work that should be developed to
address some of the issues we find. This might fall under the category of a “New approach”.
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


## System

##### Systems Overview
S

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




