import cv2
import numpy as np
import os
from os.path import isfile, join
from toolbox.makedir import make

def imagesToVideo(imageFolder,saveName,fps):
    pathIn= imageFolder
    make(join(imageFolder,"video"))
    pathOut = join(imageFolder,"video",'{}.avi'.format(saveName))
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort()
    for i,file in enumerate(files):
        filename=join(pathIn,file)
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        # inserting the frames into an image array
        frame_array.append(img)
    
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        print(i)
        out.write(frame_array[i])
    out.release()
