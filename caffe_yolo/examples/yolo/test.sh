#!/usr/bin/env sh

CAFFE_HOME=../..

PROTO=./gnet_test.prototxt
MODEL=/home/softnautics/githubyolo/caffe-yolo/examples/yolo/models/gnet_yolo_iter_80000.caffemodel
ITER=500

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL

