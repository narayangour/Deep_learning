#!/usr/bin/env sh

CAFFE_ROOT=../..
ROOT_DIR=/home/softnautics/vocdataset/
LABEL_FILE=$CAFFE_ROOT/data/yolo/label_map.txt

# 2007 + 2012 trainval
LIST_FILE=$CAFFE_ROOT/data/yolo/trainval.txt
LMDB_DIR=./lmdb3/trainval_lmdb


# 2007 test
#LIST_FILE=$CAFFE_ROOT/data/yolo/test_2007.txt
#LMDB_DIR=./lmdb/test2007_lmdb

dirname=lmdb3

if [ ! -d "$dirname" ]
then
    echo "File doesn't exist. Creating now" $dirname "folder in current directory"
    mkdir ./$dirname
    echo "File created"
else
    echo "File exists"
fi


SHUFFLE=true

RESIZE_W=224
RESIZE_H=224

$CAFFE_ROOT/build/tools/convert_box_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE $ROOT_DIR $LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

