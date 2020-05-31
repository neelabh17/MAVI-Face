import os
from os.path import join
from toolbox.pickleOpers import loadup,save
import cv2
import matplotlib.pyplot as plt
from improveDataset import *
from widerface_evaluate.bbox import bbox_overlaps
import pickle 

import numpy as np
def putbbox(img_raw,dets,mode=0,gt=None):
    '''
    mode 0 for exact mode in dets : x1, y1, x2, y2 for pred box
    mode 1 for relative mode in dets: x1, y1, w, h for gts
    '''
    new_annot=np.array([]).reshape(0,20) 
    faltuAnnot=np.array([-1.0]*16).reshape(1,16)  
    if(dets.shape[0]==0):
        return img_raw,np.array([]).reshape(0,20)
    
    if(mode==0):
            if(gt.shape[0]==0):
                print("okokokokokok this is happening")
                maxer=np.zeros((dets.shape[0],1))
            else:
                
                # plt.clf()
                # plt.imshow(img_raw[:,:,::-1])
                # plt.show()
                gt=gt.astype(float)
                _gt = gt.copy()
                _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
                _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
                dets=dets.astype(float)
                print(_gt.shape,dets.shape)
                over=(bbox_overlaps(dets[:,:4],_gt[:,:4]))
                maxer=np.max(over,axis=1)
                print(maxer.shape)
                print(maxer)
    
    for i,b in enumerate(dets):
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        if(mode==1):
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2]+b[0], b[3]+b[1]), (0, 255, 0), 2)

        cx = b[0]
        cy = b[1] + 12
        if(mode==1):
            cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        
        if(mode==0):
            if(maxer[i]<=0.3):
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            
                cv2.imshow("Test now",img_raw)
                k=cv2.waitKey(0)
                if(k==121): # yes or y
                    new_annot=np.concatenate((new_annot,np.concatenate((np.array([b[0],b[1],b[2]-b[0],b[3]-b[1]]).reshape(1,4),faltuAnnot),axis=1)),axis=0)
                cv2.destroyAllWindows()
        
    return img_raw,new_annot

def test():

    a=loadup(join("data","widerface","train","label.pickle"))
    presaved={}
    for i,fileName in enumerate(a):
        c,b=os.path.split(fileName)
        # if(os.path.isfile("C:\\Users\\neela\Desktop\Chetan\MAVI-Face\data\widerface\\train\extraAnno.pickle")):
        #     presaved=loadup("C:\\Users\\neela\\Desktop\Chetan\MAVI-Face\data\widerface\\train\extraAnno.pickle")
        # else:
        #     presaved={}
        print("{}th file".format(i))
        tp=join(join("data","widerface","train","images"),c,b)
        gtannoimg,_=putbbox(cv2.imread(tp),a[fileName],mode=1)
        inferedDets=infer(net,cv2.imread(tp))
        _,annot=putbbox(gtannoimg,inferedDets,mode=0,gt=a[fileName])
        presaved[fileName]=annot
        print(annot)
        save(presaved,"C:\\Users\\neela\Desktop\Chetan\MAVI-Face\data\widerface\\train\extraAnno.pickle")
        # input()
