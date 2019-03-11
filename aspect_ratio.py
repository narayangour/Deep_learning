#######################################################################################################
# folder_path = /Paste/Path/of/your/directory/which/contain/all/images 	   	 	
# Comannd - python aspect_ratio.py
# Output  - aspect_ratio.txt
#######################################################################################################

import os, os.path, shutil,sys
import io
from PIL import Image


folder_path = "/home/softnautics/narayan/OIDv4_ToolKit-master/OID/Dataset/train/Person/"
images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
f = open('/home/softnautics/narayan/OIDv4_ToolKit-master/'+'aspect_ratio.txt' , 'a')

min_h = 0
min_w = 0
max_h = 0
max_w = 0
flag = 0

for image in images:
    path = folder_path+image
    img = Image.open(path)
    if flag==0:
	min_h = img.size[1]
	min_w = img.size[0]
	max_h = img.size[1]
	max_w = img.size[0]
	flag = 1
	print "min_h min_w max_h max_w",min_h,min_w,max_h,max_w
    if(min_w>img.size[0]):
	min_w = img.size[0]
    if(min_h>img.size[1]):
	min_h = img.size[1]
    if(max_w<img.size[0]):
	max_w = img.size[0]
    if(max_h<img.size[1]):
	max_h = img.size[1]
    f.write(str(image)+' ')
    f.write(str(img.size))
    f.write('\n')
f.write('min_w -'+str(min_w))
f.write('\n')
f.write('min_h -'+str(min_h))
f.write('\n')
f.write('max_w -'+str(max_w))
f.write('\n')
f.write('max_h -'+str(max_h))
f.write('\n')
print "min_h min_w max_h max_w",min_h,min_w,max_h,max_w








