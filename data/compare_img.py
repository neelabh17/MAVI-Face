import sys
sys.path.append("..")
import numpy as np
import cv2
import os
from utils.evalResults import readData, reductionProcedures
def saveImages(trained_model_name,nms_threshold,vis_thresh):
    # load dataset ground truth
    fileName="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/data/widerface/val/label.pickle"
    # fileName="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/data/widerface/train/label.pickle"
    gts=readData(fileName)

    #load the predbbooxes dataset ground truth
    evalDataFolder="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/evalData/"
    fileName=evalDataFolder+trained_model_name+"/outResults.pickle"
    preds=readData(fileName)

    imageFolder="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/data/widerface/val/images/"
    # imageFolder="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/data/widerface/train/images/"
    for j,fileName in enumerate(gts):
        # print(i,fileName)
        # i=j+527 #for train set
        i=j #for val set
        #reading the images
        image_path = imageFolder + fileName
        # print("image path is :",image_path)
        img_raw_gt = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_raw_pred = cv2.imread(image_path, cv2.IMREAD_COLOR)


        #get gt data for the image
        gt_boxesToSend=np.array(gts[fileName])
        gt_boxesToSend=gt_boxesToSend[...,:4]
        gt_boxesToSend=gt_boxesToSend.astype(int)

        #print boxes on gt image
        for b in gt_boxesToSend:
            b = list(map(int, b))
            cv2.rectangle(img_raw_gt, (b[0], b[1]), (b[2]+b[0], b[3]+b[1]), (0, 0, 255), 2)
    
        
        #get pred data for image
        pred_data=preds[fileName]
        dets,predbox=reductionProcedures(pred_data,nms_threshold,vis_thresh)

        #putting up pred bbox
        for b in dets:
            if b[4] < vis_thresh:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw_pred, (b[0], b[1]), (b[2]+b[0], b[3]+b[1]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw_pred, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw_pred, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw_pred, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw_pred, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw_pred, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw_pred, (b[13], b[14]), 1, (255, 0, 0), 4)
        
        #concatinatinf the images
        c=np.concatenate((img_raw_pred,img_raw_gt),axis=0)
        image=c
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        # org 
        org = (20, 50) 
        org2 = (50, 50+c.shape[0]//2) 
        
        # fontScale 
        fontScale = 1
        
        # Blue color in BGR 
        color = (0, 0, 255) 
        color2 = (0, 0, 255) 
        
        # Line thickness of 2 px 
        thickness = 2
        # c=cv2.putText(image, '(Train Im)Pred at conf-thres :{:.4f}'.format(vis_thresh), org, font, fontScale, color, thickness, cv2.LINE_AA) 
        c=cv2.putText(image, '(Val Img)Pred at conf-thres :{:.4f}'.format(vis_thresh), org, font, fontScale, color, thickness, cv2.LINE_AA) 
        c=cv2.putText(image, 'Ground Truth', org2, font, fontScale, color, thickness, cv2.LINE_AA) 

        #saving
        savedir="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/errorAnal/images_for_comparision/"+trained_model_name
        
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        cv2.imwrite(savedir+"/{}.jpg".format(i),c)
        print("{}-th image done saving".format(i))

        


