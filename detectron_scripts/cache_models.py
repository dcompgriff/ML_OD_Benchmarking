from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import json
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
                        '--cfg',
                        dest='cfg',
                        help='cfg model file (/path/to/model_config.yaml)',
                        default=None,
                        type=str
                    )
    parser.add_argument(
                        '--wts',dest='weights',
                            help='weights model file (/path/to/model_weights.pkl)',
                            default=None,
                            type=str
                        )
    parser.add_argument(
                                '--output-dir',
                                dest='output_dir',
                                help='directory for visualization pdfs (default: /tmp/infer_simple)',
                                default='/tmp/infer_simple',
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
                                        'im_or_folder', help='image or folder of images', default=None
                                    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

cfgFileList = [
    '/detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml',
    '/detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml',
    '/detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml',
    '/detectron/configs/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml',
    '/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-50-C4_2x.yaml',
    '/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml',
    '/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml',
    '/detectron/configs/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml',
    '/detectron/configs/12_2017_baselines/retinanet_R-50-FPN_2x.yaml',
    '/detectron/configs/12_2017_baselines/retinanet_R-101-FPN_2x.yaml',
    '/detectron/configs/12_2017_baselines/retinanet_X-101-64x4d-FPN_2x.yaml'
]

weightsFileList = [
    'https://s3-us-west-2.amazonaws.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
    'https://s3-us-west-2.amazonaws.com/detectron/35857389/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_2x.yaml.01_37_22.KSeq0b5q/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
    'https://s3-us-west-2.amazonaws.com/detectron/35857952/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml.01_39_49.JPwJDh92/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
    'https://s3-us-west-2.amazonaws.com/detectron/35858198/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml.01_41_46.CX2InaoG/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
    'https://s3-us-west-2.amazonaws.com/detectron/35858828/12_2017_baselines/e2e_mask_rcnn_R-50-C4_2x.yaml.01_46_47.HBThTerB/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
    'https://s3-us-west-2.amazonaws.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
    'https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
    'https://s3-us-west-2.amazonaws.com/detectron/35859745/12_2017_baselines/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml.02_00_30.ESWbND2w/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
    'https://s3-us-west-2.amazonaws.com/detectron/36768677/12_2017_baselines/retinanet_R-50-FPN_2x.yaml.08_30_38.sgZIQZQ5/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl',
    'https:/p/s3-us-west-2.amazonaws.com/detectron/36768840/12_2017_baselines/retinanet_R-101-FPN_2x.yaml.08_33_29.grtM0RTf/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl',
    'https://s3-us-west-2.amazonaws.com/detectron/36768907/12_2017_baselines/retinanet_X-101-64x4d-FPN_2x.yaml.08_35_40.pF3nzPpu/output/train/coco_2014_train:coco_2014_valminusminival/retinanet/model_final.pkl'
]

def main(args):
    logger = logging.getLogger(__name__)
    # For each config, download and cache the model weights
    # for the docker container, so they don't have to be downloaded
    # every time.
    #for i, cfgFile in enumerate(cfgFileList):
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
