import os
from os.path import join
from toolbox.pickleOpers import loadup,save
import cv2
import matplotlib.pyplot as plt
from improveDataset import *
from widerface_evaluate.bbox import bbox_overlaps
import pickle 
from toolbox.makedir import make

import numpy as np
def getFPbbox(dets,gt=None):
    '''
    mode 0 for exact mode in dets : x1, y1, x2, y2 for pred box
    mode 1 for relative mode in dets: x1, y1, w, h for gts
    '''
    new_annot=np.array([]).reshape(0,20) 
    faltuAnnot=np.array([-1.0]*16).reshape(1,16)  
    if(dets.shape[0]==0):
        return dets
    if(gt.shape[0]==0):
        return dets
    
    
    gt=gt.astype(float)
    _gt = gt.copy()
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
    dets=dets.astype(float)
    print(_gt.shape,dets.shape)
    over=(bbox_overlaps(dets[:,:4],_gt[:,:4]))
    maxer=np.max(over,axis=1)
    newdets=np.array([]).reshape(0,dets.shape[1])
    # print(maxer.shape)
    # print(maxer)
    print(newdets.shape)
    for i,b in enumerate(dets):
        if(maxer[i]<=0.15):
            newdets=np.concatenate((newdets,b.copy().reshape(1,-1)),axis=0)

    return newdets

def getFNbbox(dets,gt=None):
    '''
    mode 0 for exact mode in dets : x1, y1, x2, y2 for pred box
    mode 1 for relative mode in dets: x1, y1, w, h for gts
    '''
    if(gt.shape[0]==0):
        return gt
    if(dets.shape[0]==0):
        return gt
    
    gt=gt.astype(float)
    _gt = gt.copy()
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
    dets=dets.astype(float)
    print(_gt.shape,dets.shape)
    over=(bbox_overlaps(dets[:,:4],_gt[:,:4]))
    over=over.T
    maxer=np.max(over,axis=1)
    newdets=np.array([]).reshape(0,gt.shape[1])
    # print(maxer.shape)
    # print(maxer)
    print(newdets.shape)
    
    for i,b in enumerate(gt):
        if(maxer[i]<=0.15):
            newdets=np.concatenate((newdets,b.copy().reshape(1,-1)),axis=0)
    return newdets
def putbbox(img_raw,dets,mode=0,gt=None):
    '''
    mode 0 for exact mode in dets : x1, y1, x2, y2 for pred box
    mode 1 for relative mode in dets: x1, y1, w, h for gts
    '''
    for i,b in enumerate(dets):
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        if(mode==1):
            if(b[2]*b[3]>=225):
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2]+b[0], b[3]+b[1]), (0, 255, 0), 2)

        cx = b[0]
        cy = b[1] + 12
    
        # cv2.putText(img_raw, text, (cx, cy),
        #         cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        
        if(mode==0):
            if((b[2]-b[0])*(b[3]-b[1])>=225):
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        
    return img_raw

def test():

    a=loadup(join("data","widerface","val","label.pickle"))

    netBase=getNet("Resnet50_Final")
    netFTFinal=getNet("new_1xOhem_shuffle_true_scheduler_e2_epoch_22")
    netFT=getNet("SingleSamplingOhemAdamLRe3_epoch_32")
    for i,fileName in enumerate(a):
        c,b=os.path.split(fileName)
        # if(os.path.isfile("C:\\Users\\neela\Desktop\Chetan\MAVI-Face\data\widerface\\train\extraAnno.pickle")):
        #     presaved=loadup("C:\\Users\\neela\\Desktop\Chetan\MAVI-Face\data\widerface\\train\extraAnno.pickle")
        # else:
        #     presaved={}
        print(fileName)
        print("{}th file".format(i))
        tp=join(join("data","widerface","val","images"),c,b)
        #adding gtboxes
        font = cv2.FONT_HERSHEY_SIMPLEX 
    
        # org 
        org = (50, 50) 
        # org2 = (50, 50+c.shape[0]//2) 
        
        # fontScale 
        fontScale = 1
        
        # Blue color in BGR 
        color = (255, 0, 0) 
        color2 = (0, 0, 255) 
        
        # Line thickness of 2 px 
        thickness = 2
        gtannoimg=putbbox(cv2.imread(tp),a[fileName],mode=1)
        gtannoimg=cv2.putText(gtannoimg, 'Ground Truth', org, font, fontScale, color, thickness, cv2.LINE_AA) 

        #adding base
        detsBase=infer(netBase,cv2.imread(tp))
        imgBase=putbbox(cv2.imread(tp),detsBase,mode=0,gt=detsBase)
        imgBase=cv2.putText(imgBase, 'Baseline', org, font, fontScale, color, thickness, cv2.LINE_AA) 
        #adding FT
        detsFT=infer(netFT,cv2.imread(tp))
        imgFT=putbbox(cv2.imread(tp),detsFT,mode=0,gt=detsFT)
        imgFT=cv2.putText(imgFT, 'Fine Tuned', org, font, fontScale, color, thickness, cv2.LINE_AA) 
        #adding FTfinal
        detsFTFinal=infer(netFTFinal,cv2.imread(tp))
        imgFTFinal=putbbox(cv2.imread(tp),detsFTFinal,mode=0,gt=detsFTFinal)
        imgFTFinal=cv2.putText(imgFTFinal, 'Fine Tuned FInal', org, font, fontScale, color, thickness, cv2.LINE_AA) 

        aim=np.concatenate((imgFTFinal,imgFT),axis=1)
        bim=np.concatenate((imgBase,gtannoimg),axis=1)
        final=np.concatenate((aim,bim),axis=0)
        folder="test/0.055"
        
        make(folder)
        cv2.imwrite(f"{folder}/{i}.jpg",final)


def testex():
    # testing excptional cases
    a=loadup(join("data","widerface","val","label.pickle"))
    netBase=getNet("Resnet50_Final")
    netFTFinal=getNet("new_1xOhem_shuffle_true_scheduler_e2_epoch_22")
    netFT=getNet("SingleSamplingOhemAdamLRe3_epoch_32")
    for i,fileName in enumerate(a):
        c,b=os.path.split(fileName)
        # if(os.path.isfile("C:\\Users\\neela\Desktop\Chetan\MAVI-Face\data\widerface\\train\extraAnno.pickle")):
        #     presaved=loadup("C:\\Users\\neela\\Desktop\Chetan\MAVI-Face\data\widerface\\train\extraAnno.pickle")
        # else:
        #     presaved={}
        print(fileName)
        print("{}th file".format(i))
        tp=join(join("data","widerface","val","images"),c,b)
        #adding gtboxes
        font = cv2.FONT_HERSHEY_SIMPLEX 
    
        # org 
        org = (50, 50) 
        # org2 = (50, 50+c.shape[0]//2) 
        
        # fontScale 
        fontScale = 1
        
        # Blue color in BGR 
        color = (255, 0, 0) 
        color2 = (0, 0, 255) 
        
        # Line thickness of 2 px 
        currImg=cv2.imread(tp)
        thickness = 2
        gtanno=a[fileName][np.where(np.multiply(a[fileName][:,2],a[fileName][:,3])>=225)[0]]
        gtannoimg=putbbox(currImg.copy(),gtanno,mode=1)
        gtannoimg=cv2.putText(gtannoimg, 'Ground Truth', org, font, fontScale, color, thickness, cv2.LINE_AA) 

        #adding base
        detsBase=infer(netBase,currImg.copy())
        detsBaseex=getFNbbox(detsBase,gtanno)
        if(detsBaseex.shape[0]>0):
            imgBase=putbbox(currImg.copy(),detsBaseex,mode=1,gt=detsBase)
            imgBase=cv2.putText(imgBase, 'Baseline', org, font, fontScale, color, thickness, cv2.LINE_AA) 
            folder="test/Baseline/FN"
            make(folder)
            print("Writing a file")
            cv2.imwrite(f"{folder}/{i}.jpg",np.concatenate((imgBase,gtannoimg),axis=1))
        detsBaseex=getFPbbox(detsBase,gtanno)
        if(detsBaseex.shape[0]>0):
            imgBase=putbbox(currImg.copy(),detsBaseex,mode=0,gt=detsBase)
            imgBase=cv2.putText(imgBase, 'Baseline', org, font, fontScale, color, thickness, cv2.LINE_AA) 
            folder="test/Baseline/FP"
            make(folder)
            print("Writing a file")
            cv2.imwrite(f"{folder}/{i}.jpg",np.concatenate((imgBase,gtannoimg),axis=1))
        #adding FT
        detsFT=infer(netFT,currImg.copy())
        detsFTex=getFNbbox(detsFT,gtanno)
        if(detsFTex.shape[0]>0):
            imgFT=putbbox(currImg.copy(),detsFTex,mode=1,gt=detsFT)
            imgFT=cv2.putText(imgFT, 'FTline', org, font, fontScale, color, thickness, cv2.LINE_AA) 
            folder="test/FTline/FN"
            make(folder)
            print("Writing a file")
            cv2.imwrite(f"{folder}/{i}.jpg",np.concatenate((imgFT,gtannoimg),axis=1))
        detsFTex=getFPbbox(detsFT,gtanno)
        if(detsFTex.shape[0]>0):
            imgFT=putbbox(currImg.copy(),detsFTex,mode=0,gt=detsFT)
            imgFT=cv2.putText(imgFT, 'FTline', org, font, fontScale, color, thickness, cv2.LINE_AA) 
            folder="test/FTline/FP"
            make(folder)
            print("Writing a file")
            cv2.imwrite(f"{folder}/{i}.jpg",np.concatenate((imgFT,gtannoimg),axis=1))
        #adding FTFinal
        detsFTFinal=infer(netFTFinal,currImg.copy())
        detsFTFinalex=getFNbbox(detsFTFinal,gtanno)
        if(detsFTFinalex.shape[0]>0):
            imgFTFinal=putbbox(currImg.copy(),detsFTFinalex,mode=1,gt=detsFTFinal)
            imgFTFinal=cv2.putText(imgFTFinal, 'FTFinalline', org, font, fontScale, color, thickness, cv2.LINE_AA) 
            folder="test/FTFinalline/FN"
            make(folder)
            print("Writing a file")
            cv2.imwrite(f"{folder}/{i}.jpg",np.concatenate((imgFTFinal,gtannoimg),axis=1))
        detsFTFinalex=getFPbbox(detsFTFinal,gtanno)
        if(detsFTFinalex.shape[0]>0):
            imgFTFinal=putbbox(currImg.copy(),detsFTFinalex,mode=0,gt=detsFTFinal)
            imgFTFinal=cv2.putText(imgFTFinal, 'FTFinalline', org, font, fontScale, color, thickness, cv2.LINE_AA) 
            folder="test/FTFinalline/FP"
            make(folder)
            print("Writing a file")
            cv2.imwrite(f"{folder}/{i}.jpg",np.concatenate((imgFTFinal,gtannoimg),axis=1))
        
