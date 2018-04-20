'''
This script accepts a path to a directory of images, and applies transformations
to all of the images in that directory, and saves the resulting images to a new
sub-directory called "transformed/".
'''
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import argparse
import os
import sys
import glob
import shutil
from multiprocessing import Pool

import imgaug as ia
from imgaug import augmenters as iaa

# Set initial pylab parameters.
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
GLOBAL_ARGS = None

# Define transform sequence objects.
transformerList = []
transformerNameList = []

# Add gaussian blurs
for mSigma in [1, 3, 10, 20]:
    transformerList.append(iaa.Sequential([
        iaa.GaussianBlur(sigma=(mSigma))  # blur images with a sigma of 0 to 3.0
    ]))
    transformerNameList.append('gaussianblur_' + str(mSigma).replace('.','p') + '__')
# Add super pixel transforms
for pReplace in [0.1, 0.5, 0.85]:
    transformerList.append(iaa.Sequential([
        iaa.Superpixels(p_replace=pReplace, n_segments=128)
    ]))
    transformerNameList.append('superpixels_' + str(pReplace).replace('.','p') + '__')

def parse_args():
    parser = argparse.ArgumentParser(description='Image Transformer')
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for transformed images.',
        default='/tmp/transformed',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--num-cores',
        dest='num_cores',
        help='number of cores to use when performing transformations in parallel.',
        default=8,
        type=int
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def buildTransformerList():
    # Define transform sequence objects.
    transformerList = []
    transformerNameList = []

    # Add gaussian blurs
    for mSigma in [1, 3, 10, 20]:
        transformerList.append(iaa.Sequential([
            iaa.GaussianBlur(sigma=(mSigma))  # blur images with a sigma of 0 to 3.0
        ]))
        transformerNameList.append('gaussianblur_' + str(mSigma).replace('.', 'p') + '__')
    # Add super pixel transforms
    for pReplace in [0.1, 0.5, 0.85]:
        transformerList.append(iaa.Sequential([
            iaa.Superpixels(p_replace=pReplace, n_segments=64)
        ]))
        transformerNameList.append('superpixels_' + str(pReplace).replace('.', 'p') + '__')

    return transformerList, transformerNameList

# Apply all image transforms for the given image.
def applyAllTransforms(im_name):
    display = False

    # Get the base name for the image.
    print('Transforming %s' % im_name)
    baseImgName = os.path.basename(im_name)
    I = io.imread(im_name)

    # Apply the transforms in the transformer list.
    for i, seq in enumerate(transformerList):
        IaugName = transformerNameList[i] + baseImgName
        Iaug = seq.augment_image(I)
        io.imsave(GLOBAL_ARGS.output_dir + IaugName, Iaug)

        if display:
            plt.axis('off')
            plt.imshow(Iaug)
            plt.show()

def main(args):
    # Get the image paths to the specified input folder.
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    # Check for existence of ouput dir, or make it.
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # For each image, apply the set of transforms, generate new images, and output them to
    # the specified directory.
    for i, im_name in enumerate(im_list):
        applyAllTransforms(im_name)











if __name__ == '__main__':
    args = parse_args()
    GLOBAL_ARGS = args
    main(args)











