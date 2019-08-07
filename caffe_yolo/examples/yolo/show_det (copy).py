#!/usr/bin/env python

#  python2 show_det.py

import time 
import numpy as np
import re
import cv2
import os, sys
sys.path.insert(0, '../../python/')
import caffe
#caffe.set_device(0)
caffe.set_mode_cpu()
np.set_printoptions(threshold=np.inf)
mean = np.require([104, 117, 123], dtype=np.float32)[:, np.newaxis, np.newaxis]

def nms(dets, thresh):
  # -------------------------
  # Pure Python NMS baseline.
  # Written by Ross Girshick
  # -------------------------
  x1 = dets[:, 0] - dets[:, 2] / 2.
  y1 = dets[:, 1] - dets[:, 3] / 2.
  x2 = dets[:, 0] + dets[:, 2] / 2.
  y2 = dets[:, 1] + dets[:, 3] / 2.
  scores = dets[:, 4]
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]
  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas[i] + areas[order[1:]] - inter)
    inds = np.where(ovr <= thresh)[0]
    order = order[inds + 1]
  return dets[np.require(keep), :]
minprob = 1
def parse_result(reg_out):
  # update num_classes variable as per your custom dataset
  num_classes = 3
  num_objects = 2
  side = 7
  locations = side ** 2
  #print("locations-",locations,num_objects)
  boxes = np.zeros((num_objects * locations, 6), dtype=np.float32)
  #print ('obj_scores & location')
  for i in range(locations):
    tmp_scores = reg_out[i:num_classes*locations:locations]
    #print 'tmp_scores of Probablaites+++++++++++++ ',i,num_classes*locations,locations
    #print (tmp_scores,i)
    max_class_ind = np.argsort(tmp_scores)[-1]
    #print 'max_class_ind'
    #print max_class_ind
    max_prob = np.max(tmp_scores)
    #print 'max_prob'
    #print max_prob
    #global minprob
    #if minprob>max_prob:
      #minprob = max_prob
    #print "minprob->",minprob

    ######## Probs over, start for confidance
    obj_index = num_classes * locations + i
    obj_scores = max_prob * reg_out[obj_index:(obj_index+num_objects*locations):locations]
#   print 'CONFIDANCE+++++++++++++++++++++++++++++++++++'
    #print (reg_out[obj_index:(obj_index+num_objects*locations):locations],i)
#   print 'obj_scores & location'
    #print (obj_scores,i)

    ######## Objs over, Coordinates start

    coor_index = (num_classes + num_objects) * locations + i
    for j in range(num_objects):
      boxes[i*num_objects+j][5] = max_class_ind
      boxes[i*num_objects+j][4] = obj_scores[j]
      box_index = coor_index + j * 4 * locations
      boxes[i*num_objects+j][0] = (i % side + reg_out[box_index + 0 * locations]) / float(side)
      boxes[i*num_objects+j][1] = (i / side + reg_out[box_index + 1 * locations]) / float(side)
      boxes[i*num_objects+j][2] = reg_out[box_index + 2 * locations] ** 2
      boxes[i*num_objects+j][3] = reg_out[box_index + 3 * locations] ** 2
     # print'===========-=-=-=-=--=-=-=-=-====================-=-=-=-=-========='
     # print 'value of x at loaction'
     # print boxes[i*num_objects+j][0],reg_out[box_index + 0 * locations],i,box_index
     # print 'value of y'
     # print  boxes[i*num_objects+j][1],reg_out[box_index + 1 * locations],i
     # print 'value of w'
     # print boxes[i*num_objects+j][2],reg_out[box_index + 2 * locations],i
     # print 'value of h'
     # print boxes[i*num_objects+j][3],reg_out[box_index + 3 * locations],i
     # print'===========-=-=-=-=--=-=-=-=-====================-=-=-=-=-========='
  return nms(boxes, 0.7)

def show_boxes(im_path, boxes, thresh=0.5, show=0):
  #print (boxes.shape)
  print "thresh - ",thresh
  out_path = "./test2/out/"
  is_class='dummy'
  im = cv2.imread(im_path)
  ori_w = im.shape[1]
  ori_h = im.shape[0]
  for box in boxes:
    #print("box[4]- ",box[4])
    if box[4] < thresh:
      continue
    #print ('+++++++++++++++++++++selected++++++++++++++++++++++')
    print "obj_scores -",box[4]

    # update condition as per label_map.txt file
    if box[5] == 0:
      is_class='bird'
    if box[5] == 1:
      is_class='dog'
    if box[5] == 2:
      is_class='horse'
    print "Predicted class name -",is_class

    time.sleep(3)
    #print ('+++++++++++++++++++++selected++++++++++++++++++++++')
    box = box[:4]
    #print ('printing x,y,w,h')
    #print (box[0])
    #print (box[1])
    #print (box[2])
    #print (box[3])
    #print ('printing original width & hight')
    #print (ori_w)
    #print (ori_h)
    x1 = max(0, int((box[0] - box[2] / 2.) * ori_w))
    #print ('max of x1')
    #print ((box[0] - box[2] / 2.) * ori_w)
    y1 = max(0, int((box[1] - box[3] / 2.) * ori_h))
    #print ('max of y1')
    #print ((box[1] - box[3] / 2.) * ori_h)
    x2 = min(ori_w - 1, int((box[0] + box[2] / 2.) * ori_w))
    #print ('min of x2')
    #print ((box[0] + box[2] / 2.) * ori_w)
    y2 = min(ori_h - 1, int((box[1] + box[3] / 2.) * ori_h))
    #print ('min of y2')
    #print ((box[1] + box[3] / 2.) * ori_h)
    #print (x1,y1,x2,y2)
    #print (box[2])
    #print (box[3])
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 100
    fontColor              = (25,55,255)
    lineType               = 6
    
    cv2.putText(im,is_class, 
    (int (x2-100),int(y2-100)),2,1.5,(0,0,255),
    )
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
  if show:
    cv2.imshow("out", im)
    cv2.waitKey(0)
  else:
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    writeStatus = cv2.imwrite(out_path+im_path.split('/')[-1], im)
    if writeStatus is True:
      print "image written at path: ",out_path
      print "="*40
    else:
      print("problem in image writting")

  

def det(model, im_path, show=0):
  '''forward processing'''
  im = cv2.imread(im_path)
  im = cv2.resize(im, (224, 224))
  im = np.require(im.transpose((2, 0, 1)), dtype=np.float32)
  im -= mean
  model.blobs['data'].data[...] = im
  out_blobs = model.forward()
  #print("out_blobs -",out_blobs)
  reg_out = out_blobs["regression"]
  #reg_out = out_blobs["result"]
  #print (reg_out)
  #print ('whole dump')
  #print (reg_out[0])
  boxes = parse_result(reg_out[0])
  show_boxes(im_path, boxes, 0.07)

def load_test_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
      images.append(os.path.join(folder, filename))
    return images




if __name__=="__main__":
  # create test2 folder and under test2 again create images folder, copy all test images into images folder
  im_folder_path = "./test2/images/"
  test_img_list = load_test_images_from_folder(im_folder_path)
  print("test_img_list -",test_img_list)

  # copy Network file into test2 folder 
  net_proto = "./test2/narayan_inference.proto"

  # copy model file into test2 folder 
  model_path = "./test2/gnet_yolo_iter_19000.caffemodel"
  model = caffe.Net(net_proto, model_path, caffe.TEST)

  for testImg in test_img_list:
    print "Input Img path -",testImg
    det(model, testImg, show=1)







