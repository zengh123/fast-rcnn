#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb,list_imdbs
import caffe
import argparse
import pprint
import numpy as np
import sys
import os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)

    # -----------
    #code below is only used in debug. By setting number after > and <, we can select
    #part of dataset and makes the debug process faster
    k_temp = len(imdb._image_index)
    print k_temp    
    if (args.imdb_name == 'coco_train2014'):
        for i in range(0,k_temp):
            if imdb._image_index[k_temp - i -1] < 0:
                 del imdb._image_index[k_temp- i -1]
    else:
       for i in range(0,k_temp):
           if int(imdb._image_index[k_temp - i -1]) < 0:         
               print imdb._image_index[k_temp- i -1]
               del imdb._image_index[k_temp- i -1]
           elif int(imdb._image_index[k_temp - i -1]) > 20000:
               print imdb._image_index[k_temp- i -1]
               del imdb._image_index[k_temp- i -1]
    print 'imdb._image_index : new len after picking up:'
    print len(imdb._image_index)
    print ''
    # -----------

    output_dir = get_output_dir(imdb, None)
    roidb = get_training_roidb(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    print 'max iter in train_net'
    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
