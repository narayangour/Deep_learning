#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=./gnet_solver.prototxt

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --snapshot=/home/softnautics/githubyolo/caffe-yolo/examples/yolo/models/gnet_yolo_iter_49000.solverstate

