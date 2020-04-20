import cv2
import numpy as np
import os
from os.path import isfile, join
pathIn= '/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/errorAnal/images_for_comparision/Resnet50_epoch_28_noGrad_FT_Adam_lre3/val-new_annot_remove_sm/merge/'
pathOut = 'video.mp4'
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#for sorting the file names properly
files.sort()
for i in range(len(files)):
    print("ok{}".format(i))
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
fps = 1
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    print(i)
    out.write(frame_array[i])
out.release()