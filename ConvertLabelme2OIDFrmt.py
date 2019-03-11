######################################################################
# command - python ConvertLabelme2OIDFrmt.py
# Note : Update Path As per your Folder structure
# Description - if class name is "person" in labelImg tool, then it 
#		will read labels of labelImg tool annotated and 
#		convert into Open Image Dataset format and saved
#		into "write_fold" folder path
# Input - 1) Images Path
#	  2) Labels Path
# Output - Open Image Dataset Format Labels 
######################################################################

import csv
import sys,os
import progressbar
import cv2

Label2path = '/home/narayan/narayan/Label_2/'			# Path where all the Labels are present
Image2path = '/home/narayan/narayan/Image_2/'			# Path where all the Images are present 
write_fold = '/home/narayan/narayan/fp_KITTI/'			# Path where converted output Labels are stored


def ConvertLabelme2OID(RootFolder, PrefixFolderPath = False):
    if len(RootFolder.strip()) == 0 or not os.path.exists(RootFolder):
           raise ValueError('Error: ConvertLabelme2OID(): RootFolder path '\
                            'empty or invalid - "' + RootFolder + '"')
    TxtFileList = []
    localTxtFileList = []
    ImgFileList = []
    match_cnt = 0
    not_match_cnt = 0

    print "Path where all the Labels are present - ", RootFolder
    for rootdir,subdirs,files in os.walk(RootFolder):
	for filename in files:
	    TxtFileList.append(rootdir+'/'+filename)
    TxtFileList = set(TxtFileList)
    print "Total Labels are - ",len(TxtFileList)



    print "Path where all the Images are present - ", Image2path
    for rootdir,subdirs,files in os.walk(Image2path):
	for filename in files:
	    ImgFileList.append(rootdir+'/'+filename)
    print "Total Images are  - ",len(ImgFileList)
    print "Path where converted output Labels are stored -",write_fold


    bar = progressbar.ProgressBar(maxval=len(TxtFileList),widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    i=0
    flag = 0
    for txtName in TxtFileList:
	i = i+1
	for Imgname in ImgFileList:
	    if str(Imgname.split('/')[-1].split('.')[0]) == str(txtName.split('/')[-1].split('.')[0]):
		match_cnt = match_cnt + 1
		flag = 1
                img = cv2.imread(Imgname)		# height, width, channels
		if img is not None:
	                if os.path.isfile(txtName):
	                    read = open(txtName, 'r')
			    write = open(write_fold+txtName.split('/')[-1], 'w')
			    for line in read:
        			if line.split()[0]=='1':
        			    narayan = line.split()
        			    narayan[0] = 'Person'
				    Ymax = float(2)*float(img.shape[0])*float(narayan[2]) + float(narayan[4])*float(img.shape[0])
				    Ymax = float(Ymax/2)
                                    if Ymax>img.shape[0]:
                                        Ymax = img.shape[0]			
				    #print("Ymax - ",Ymax)
				    Ymin = float(2)*float(img.shape[0])*float(narayan[2])-float(Ymax)
				    #print("Ymin - ",Ymin)
				    Xmax = float(2)*float(img.shape[1])*float(narayan[1]) + float(narayan[3])*float(img.shape[1])
				    Xmax = float(Xmax/2)
                                    if Xmax>img.shape[1]:
                                        Xmax = img.shape[1]			
				    #print("Xmax - ",Xmax)
				    Xmin = float(Xmax) - float(img.shape[1])*float(narayan[3])
				    #print("Xmin - ",Xmin)
            			    narayan[1] = str(Xmin)
            			    narayan[2] = str(Ymin)
            			    narayan[3] = str(Xmax)
            			    narayan[4] = str(Ymax)
			            s=' '
        			    s = s.join(narayan) 
        			    line=s
				    write.writelines(line)
				    write.writelines('\n')				
			    read.close()
			    write.close()
	                else:
	                    print txtName.split('/')[-1],"is not text file"
	
		else:
			print "Failed to read image name ",Imgname.split('/')[-1]
	if flag!=1:
		not_match_cnt = not_match_cnt + 1
		flag=0
        bar.update(i)
    print "Labels Does not match with respective image -",not_match_cnt
    print "Tatal Labels and Images match cnt -",match_cnt
    bar.finish()

ConvertLabelme2OID(Label2path)












