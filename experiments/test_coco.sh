#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/`basename $0`.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"




time ./tools/test_net.py --gpu 0 \
  --def models/CaffeNet/test.prototxt \
  --net output/default/coco_train2014/caffenet_fast_rcnn_iter_20000.caffemodel \
  --imdb coco_val2014

