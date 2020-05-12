import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from utils.evalResults import readData, reductionProcedures

from scipy.io import loadmat
from widerface_evaluate.bbox import bbox_overlaps
from IPython import embed

file=open("/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/errorAnal/fp.pickle","rb")
a=pickle.load(file)
file.close()
errorList=["Blur","Small","Dark","Tree","occulusion","in group","dif object","non annotated face","less iou"]
# errorList=["Small","Dark","in group","not predicted","less iou"]



dtype=[(x,int) for x in errorList]
setList={setname: set() for setname in errorList}

totalWrongPredBoxes=0
for filename in a:
    nparray=a[filename]
    totalWrongPredBoxes+=len(np.where(np.sum(nparray,axis=1)>0)[0])
    for i , setname in enumerate(setList):
        ok=(np.where(nparray[:,i]==1)[0])
        for j in ok:
            setList[setname].add(filename+str(j))

evalDataFolder="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/evalData/"
fileName=evalDataFolder+"Resnet50_epoch_28_noGrad_FT_Adam_lre3"+"/outResults_val.pickle"
preds=readData(fileName)


b=[]
# for x in setList["Small"]:
for x in setList["non annotated face"]:
    imageName,no=x.split(".jpg")
    imageName=imageName+".jpg"
    no=int(no)
    # print(imageName,no)
    pred_data=preds[imageName]
    dets,predbox=reductionProcedures(pred_data,nms_threshold=0.5,confidence_threshold=0.055)
    b.append((dets[no][2]*dets[no][3]))

print(len(b))