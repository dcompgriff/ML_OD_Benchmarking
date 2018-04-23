'''
This script accepts a path to a directory of images, and applies transformations
to all of the images in that directory, and saves the resulting images to a new
sub-directory called "transformed/".
'''
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

# Add gaussian blurs (Focus type blurs)
for mSigma in [1, 10, 20]:
    transformerList.append(iaa.Sequential([
        iaa.GaussianBlur(sigma=(mSigma))  # blur images with a sigma of 0 to 3.0
    ]))
    transformerNameList.append('gaussianblur_' + str(mSigma).replace('.','p') + '__')
# Add super pixel transforms. (Region color and shape based sensitivity)
for pReplace in [0.1, 0.5, 0.85]:
    transformerList.append(iaa.Sequential([
        iaa.Superpixels(p_replace=pReplace, n_segments=128)
    ]))
    transformerNameList.append('superpixels_' + str(pReplace).replace('.','p') + '__')
# Add hue based transforms. (Color sensitivity testing)
for hue in [25, 50]:
    transformerList.append(iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((hue))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    ]))
    transformerNameList.append('colorspace_' + str(hue).replace('.','p') + '__')
# Add average type blurs. (Shaky camera or moving object type blurs)
for width, height in [(5, 11)]:
    transformerList.append(iaa.Sequential([
        iaa.AverageBlur(k=((width), (height)))
    ]))
    transformerNameList.append('averageblur_' + str(width) + '_' + str(height) + '__')
# Add median type blurs. (Region type blurs)
for size in [1]:
    transformerList.append(iaa.Sequential([
        iaa.MedianBlur(k=(3, 11))
    ]))
    transformerNameList.append('medianblur_' + str(size) + '__')
# Add sharpen type affects. (Lighting type issues)
for num in range(3):
    transformerList.append(iaa.Sequential([
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
    ]))
    transformerNameList.append('sharpen_' + str(num) + '__')
# Add intensity type affects. (Lighting type issues)
for num in [-80, 80]:
    transformerList.append(iaa.Sequential([
        iaa.Add((num))
    ]))
    transformerNameList.append('addintensity_' + str(num) + '__')
# Add random intensity type affects. (Intensity noise type issues)
for num in [1]:
    transformerList.append(iaa.Sequential([
        iaa.AddElementwise((-80, 80))
    ]))
    transformerNameList.append('elementrandomintensity_' + str(num) + '__')
# Add lighting intensity type affects. (Lighting type issues)
for num in [.25, 2]:
    transformerList.append(iaa.Sequential([
        iaa.Multiply((num))
    ]))
    transformerNameList.append('multiplyintensity_' + str(num).replace('.','p') + '__')
# Add contrast type affects. (Color type issues)
for num in range(3):
    transformerList.append(iaa.Sequential([
        iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)
    ]))
    transformerNameList.append('contrastnormalization_' + str(num).replace('.','p') + '__')
# Add elastic type affects. (Color, moving pixels, locality type issues) Almost like looking through a rainy windshield.
for num in [1]:
    transformerList.append(iaa.Sequential([
        iaa.ElasticTransformation(alpha=(5.0), sigma=0.25)
    ]))
    transformerNameList.append('elastic_' + str(num).replace('.','p') + '__')


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
        '--replace-output-dir',
        dest='replace_output_dir',
        help='should the output directory be replaced or not. 1=True 0=False',
        default=1,
        type=int
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

# Apply all image transforms for the given image.
def applyAllTransforms(im_name, display=False):
    # Get the base name for the image.
    print('Transforming %s' % im_name)
    baseImgName = os.path.basename(im_name)
    I = io.imread(im_name)

    # Apply the transforms in the transformer list.
    for i, seq in enumerate(transformerList):
        try:
            IaugName = transformerNameList[i] + baseImgName
            Iaug = seq.augment_image(I)
            io.imsave(GLOBAL_ARGS.output_dir + IaugName, Iaug)

            if display:
                plt.axis('off')
                plt.imshow(Iaug)
                plt.show()
        except:
            print('Error while transforming %s'%IaugName)

            
def main(args):
    # Get the image paths to the specified input folder.
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    # Check for existence of ouput dir, or make it.
    if args.replace_output_dir == 1:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
    # For each image, apply the set of transforms, generate new images, and output them to
    # the specified directory.
    for i, im_name in enumerate(im_list):
        print("Image %d"%i)
        applyAllTransforms(im_name)











if __name__ == '__main__':
    args = parse_args()
    GLOBAL_ARGS = args
    main(args)











