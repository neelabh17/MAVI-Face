"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed
from utils.evalResults import readData, reductionProcedures


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.01, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()
def neel_image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    # print(pred.shape,gt.shape)
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
# 
    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    # print(overlaps.shape)
    # input()

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            # if ignore[max_idx] == 0:
            # my change made
            if False:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list

def neelEvaluation(pred, gt_path, iou_thresh,n):
    count_face = 0
    thresh_num = 1000
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    aps=[]
    finalholder=[]
    lines=a.readlines()
    gt_boxes=[]
    totalFacesAnno=0
    i=0
    j=0
    ig=0
    ignoreimgs=[18,37,44,51,54,67,72,83,87,90,91,93,94,98,101,102,105,114,117,159,166,184,217,219,235,238,267,279,281,287,295,314,315,327,329,334,336,348]
    isFirst=True

    #load val dataset ground truth
    fileName="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/data/widerface/val/label.pickle"
    gts=readData(fileName)

    #load the predbbooxes dataset ground truth
    evalDataFolder="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/evalData/"
    fileName=evalDataFolder+args.trained_model.strip(".pth").strip("/weights/")+"/outResults.pickle"
    preds=readData(fileName)

    for fileName in gts:
        gt_boxesToSend=gts[fileName]
        pred_data=preds[fileName]
        dets,pred_boxes=reductionProcedures(pred_data)
        if(pred_boxes.shape[0]>)0 and gt_boxesToSend.shape[0]>0):
            ignore = np.zeros(gt_boxesToSend.shape[0])
            count_face+=len(gt_boxesToSend)
            pred_recall, proposal_list = neel_image_eval(predbox, gt_boxesToSend, ignore, iou_thresh)
            _img_pr_info = img_pr_info(thresh_num, predbox, proposal_list, pred_recall)
            pr_curve += _img_pr_info
    

    for line in lines:
        # print("the line is"+line)
        line=line.strip()
        if(line.startswith("#")):
            j+=1
            if isFirst:
                isFirst=False
                prevName=line[1:].strip()
            else:
                #to besned for verification beacuse these have been accumulated from earlier iterations
                gt_boxesToSend=np.array(gt_boxes.copy())
                # f=open("/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/widerface_evaluate/widerface_txt/"+prevName[:-4]+".txt","r")
                f=open(pred+prevName[:-4]+".txt","r")
                prevName=line[1:].strip()
                lines2=f.readlines()
                noPred=int(lines2[1])
                predbox=[]
                if(noPred>0):
                    for line2 in lines2[2:]:
                        line2=line2.split()
                        labelpred=[float(x) for x in line2]
                        predbox.append(labelpred)
                        
                    predbox=np.array(predbox)
                if(noPred>0 and gt_boxesToSend.shape[0]>0):
                    ignore = np.zeros(gt_boxesToSend.shape[0])
                    count_face+=len(gt_boxesToSend)
                    pred_recall, proposal_list = neel_image_eval(predbox, gt_boxesToSend, ignore, iou_thresh)
                    _img_pr_info = img_pr_info(thresh_num, predbox, proposal_list, pred_recall)
                    pr_curve += _img_pr_info

                f.close()
                # print(noPred)
                totalFacesAnno+=gt_boxesToSend.shape[0]

                #finding the prediictoin data boxed
                gt_boxes=[]
                # print(name)
                i+=1
            ig+=1

        else:
            line=line.split(" ")
            label=[float(x) for x in line]
            # print("printing the labels")
            # print(label)
            gt_boxes.append(label[:4])

    # print(i,j,totalFacesAnno)
    a.close()

    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]

    ap = voc_ap(recall, propose)
    aps.append(ap)
    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    # print("Medium Val AP: {}".format(aps[1]))
    # print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")
    return aps[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="./widerface_txt/Resnet50_Final/")
    parser.add_argument('-g', '--gt', default='./ground_truth/')

    args = parser.parse_args()

    # n=int(input("0 for NJIS,,,, 1 for Widerface:  "))
    #doing this intentionally so that i dint have to inupt it every single time
    n=0

    ious=[0.5+0.05*i for i in range(10)]
    iouVsAP=[]
    for i in ious:
        # print(i)
        # input()
        iouVsAP.append([i,neelEvaluation(args.pred,args.gt,i,n)])
    summer=0

    a=open(args.pred+"resultsremoved.txt","w")
    for itemer in iouVsAP:

        print(itemer)
        a.write(str(itemer))
        a.write("\n")
        summer+=itemer[1]

    print("=================================================")
    print("mAP is : " +str(summer/10))
    a.write("===============================================\nmAP is : "+str(summer/10))
    a.close()
    # print(neelEvaluation(args.pred, args.gt,0.3))












