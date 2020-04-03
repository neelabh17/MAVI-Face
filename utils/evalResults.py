from __future__ import print_function
import os
import cv2
import pickle
import numpy as np
import sys
from utils.nms.py_cpu_nms import py_cpu_nms



def readData(fileLocation):
    dataFile=open(fileLocation,"rb")
    data=pickle.load(dataFile)
    dataFile.close()
    # data[<filename>][<"loc">/<"landms">/<"conf">]
    return data

def reductionProcedures(imgData,nms_threshold,confidence_threshold):
    #imgData["loc"/"conf"/"landms"]
    scores=np.array(imgData["conf"])
    boxes=np.array(imgData["loc"])
    landms=np.array(imgData["landms"])
    #converting boxed to x1,y1,x2,y2 format
    boxes[...,2]+=boxes[...,0]
    boxes[...,3]+=boxes[...,1]

    # print(boxes.shape,landms.shape,scores.shape)
    # print(scores)
    # # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]
    # keep top-K faster NMS
    # dets = dets[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    #converting back to x1 y1 x2 y2 for dets to do nms but preds are still in x y w h format
    preds=dets[..., :5]
    temp=dets[...,:4].astype(int)
    preds[...,:4]=temp
    preds[...,2]-=preds[...,0]
    preds[...,3]-=preds[...,1]
    
    return dets, preds



if __name__=="__main__":

    fileLocation=args.save_folder+args.trained_model.strip(".pth").strip("/weights/")+"/outResults.pickle"
    outData=readData(fileLocation)

    testset_folder = args.dataset_folder#basically this is "./data/widerface/val/images/"

    for i,img_name in enumerate(outData):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        dets=reductionProcedures(outData[img_name])
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            imgPath=args.save_folder+args.trained_model.strip(".pth").strip("/weights/")+"/imgResults/"
            if not os.path.exists(imgPath):
                os.makedirs(imgPath)
            name = imgPath + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)

    #now calling evaluation simultaneously
    os.chdir("./widerface_evaluate/")
    str="./widerface_txt/"+args.trained_model.strip(".pth").strip("/weights/")+"/"
    os.system("python neelPipelineEvaluation2.py --pred \'{}\'".format(str))