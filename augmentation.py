# python3 augmentation.py


#!

from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
from numpy import array
import argparse
import sys
import progressbar
import shutil


ROOT_DIR      = ''
IMG_DIR       =  os.path.join(ROOT_DIR, 'images')		
LABEL_DIR     =  os.path.join(ROOT_DIR, 'Label')		
AUG_IMG_DIR   =  os.path.join(ROOT_DIR, 'aug_img')
AUG_LABEL_DIR =  os.path.join(ROOT_DIR, 'aug_label') 


class Augmentation():

    def __init__(self):
        self.imglist = []
        self.labellist = []
        self.empty_cnt = 0

    def listOfItemInDir(self,folerpath):
        if len(folerpath.strip()) == 0 or not os.path.exists(folerpath):
             raise ValueError('Error: listOfItemInDir(): folerpath path '\
                            'empty or invalid - "' + folerpath + '"')
        itemlist = []
        for rootdir,subdirs,files in os.walk(folerpath):
            for itemname in files:
                filepath = rootdir +'/'+itemname
                itemlist.append(filepath)
        return itemlist

    def AugmentationOfImg(self,prefix):
        self.imglist = self.listOfItemInDir(IMG_DIR)
        self.labellist = self.listOfItemInDir(LABEL_DIR)
        for imagepath in self.imglist:
            #print("imagename -",imagepath.split('/')[1])
            img = cv2.imread(imagepath)[:,:,::-1] #OpenCV uses BGR channels
            #print("actual shape - ",img.shape)
            if img is None:
                print ("Image is empty")
            labelpath = LABEL_DIR +'/'+imagepath.split('/')[-1].split('.')[0]+'.txt'
            try:
                read = open(labelpath)
            except:
                print ("%s open failed\n\n"%(labelpath))
            bboxes = read.readlines()

            #print("actual -",bboxes)
            read.close()
            bboxes_array = []
            for bbox in bboxes:
                bbox  = bbox.split(' ')[1:]
                bbox[3] = bbox[-1][:-1]
                bboxes_array.append([float(i) for i in bbox])
            out_bboxes_array = array(bboxes_array)
            if out_bboxes_array.size==0:
                print(i,'initial empty',labelpath)
                continue
            #plt.figure("Orginal")
            #plt.imshow(draw_rect(img, out_bboxes_array))
            #plt.pause(3) # <-------
            

            transforms = Sequence(OperationsList) 
            outimg, outbboxes = transforms(img, out_bboxes_array)

            #plt.figure("transforms")
            #plt.imshow(draw_rect(outimg, outbboxes))
            #plt.pause(3) # <-------

            if outbboxes.size==0:
                self.empty_cnt = self.empty_cnt + 1
                print(i,imagepath.split('/')[1],'- ',self.empty_cnt)
            else:
                labelnamewithpath = AUG_LABEL_DIR + '/'+prefix+imagepath.split('/')[-1].split('.')[0]+'.txt'
                #print("labelnamewithpath -",labelnamewithpath)
                try:
                    write = open(labelnamewithpath,'w+')
                except:
                    print ("%s writting failed"%(labelnamewithpath))
                for bbox in outbboxes:
                    s =' '
                    updatedlabel = ['Person']
                    for coord in bbox:
                        updatedlabel.append(str(coord))
                    s = s.join(updatedlabel) 
                    write.write(s)
                    write.write('\n')
                write.close()

                OutImgloc = AUG_IMG_DIR+'/'+prefix+imagepath.split('/')[-1]
                #print("OutImgloc -",OutImgloc)
                outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)    #  saving back in RGB channels
                #outimg = cv2.cvtColor(draw_rect(outimg, outbboxes), cv2.COLOR_BGR2RGB)    #  saving back in RGB channels with bbox drawn
                cv2.imwrite(OutImgloc,outimg) 
        if self.empty_cnt>0:
            print("Number of empty bbox file - ",self.empty_cnt)

    def CopyOrgImgLabel(self):
        org_imglist = self.listOfItemInDir(IMG_DIR)
        org_labellist = self.listOfItemInDir(LABEL_DIR)
        for imgpath in org_imglist:
            shutil.copy(IMG_DIR+'/'+imgpath.split('/')[-1], AUG_IMG_DIR+'/'+imgpath.split('/')[-1])
            shutil.copy(LABEL_DIR+'/'+imgpath.split('/')[-1].split('.')[0]+'.txt', AUG_LABEL_DIR+'/'+imgpath.split('/')[-1].split('.')[0]+'.txt')


if __name__ == '__main__':

    print("\nMenu: ")
    print(" a. RandomHorizontalFlip")
    print(" b. RandomScale")
    print(" c. RandomRotate")
    print(" d. RandomTranslate")
    print(" e. RandomShear")
    print(" f. HorizontalFlip")
    print(" g. Scale")
    print(" h. Rotate")
    print(" i. Translate")
    print(" j. Shear")
    print(" k. GaussianFiltering")
    print(" l. RandomBrightness")
    print(" m. addsnow\n\n")

    obj        = Augmentation()
    obj.CopyOrgImgLabel()

    all_op = ['a','a_b','a_b_c','a_b_c_d','a_b_c_d_e','45','90','135','180','225','270', '315','45_a','90_a','135_a','180_a','225_a','270_a', '315_a','k','l2','j','i','l0.5','m']

    manu_dict = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10, 'k':11,'l2':12,'m':13,'45':8,'90':8,'135':8,'180':8,'225':8,'270':8,'315':8,'2':12,'l0.5':12}

    optlist = ['RandomHorizontalFlip','RandomScale','RandomRotate','RandomTranslate','RandomShear','HorizontalFlip','Scale','Rotate','Translate','Shear','GaussianFiltering', 'RandomBrightness','addsnow']
    print("Running Augmentation...!!!")
    bar = progressbar.ProgressBar(maxval=len(all_op),widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    i=0
    for op_name in all_op:
        i = i + 1
        op_name = op_name.split("_")
        prefix_list=[]
        OperationsList = []
        for selection in op_name:
            if selection not in manu_dict.keys():
                print (' '*15,'>',selection,"is not valid operation.....so ignored")
                continue
            else:
                prefix_list.append(selection) 
            if manu_dict[selection]-1 ==0 or manu_dict[selection]-1==5:
                if manu_dict[selection]-1 ==0:
                    OperationsList.append(globals()[optlist[manu_dict[selection]-1]](1))
                else:
                    OperationsList.append(globals()[optlist[manu_dict[selection]-1]]())

            if manu_dict[selection]-1 ==1 or manu_dict[selection]-1==6:
                OperationsList.append(globals()[optlist[manu_dict[selection]-1]](0.2, True))
            if manu_dict[selection]-1 ==2 or manu_dict[selection]-1==7:
                if manu_dict[selection]-1==7:
                    angle = int(prefix_list[0])
                    OperationsList.append(globals()[optlist[manu_dict[selection]-1]](angle))
                else:
                    OperationsList.append(globals()[optlist[manu_dict[selection]-1]]())
            if manu_dict[selection]-1 ==3 or manu_dict[selection]-1==8:
                if manu_dict[selection]-1 ==3:
                    OperationsList.append(globals()[optlist[manu_dict[selection]-1]](0.2,False))
                else:
                    OperationsList.append(globals()[optlist[manu_dict[selection]-1]](0.2, 0.2, False))
            if manu_dict[selection]-1 ==4 or manu_dict[selection]-1==9:
                arg = 0.2
                OperationsList.append(globals()[optlist[manu_dict[selection]-1]](arg))
            if manu_dict[selection]-1 ==10:
                filter_size = 21
                OperationsList.append(globals()[optlist[manu_dict[selection]-1]](filter_size))
 
            if manu_dict[selection]-1 ==11:
                if prefix_list[0]=='l2':
                    gamma = 2
                elif prefix_list[0]=='l0.5':
                    gamma = 0.5
                OperationsList.append(globals()[optlist[manu_dict[selection]-1]](gamma))

            if manu_dict[selection]-1 ==12:
                snow_coeff = 0.5
                OperationsList.append(globals()[optlist[manu_dict[selection]-1]](snow_coeff))
        prefixImgLabName = ''.join(prefix_list)+'_'
        assert not (len(OperationsList)==0), 'Please select valid operation from above list'

        print(i)
        for op in OperationsList:
            print(" "*15,'>',str(op).split(' ')[0].split('.')[-1])
        print('\n')
        obj.AugmentationOfImg(prefixImgLabName)
        bar.update(i)
    bar.finish()
    print("\nDone Augmentation...!!!\n\n")











