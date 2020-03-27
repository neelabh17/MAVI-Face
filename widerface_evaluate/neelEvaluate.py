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


def get_iou(bbPred, bbAnno):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # print("the difference")
    # print(bbAnno,(bbPred)) 
    # input()
    assert bbPred[0] < bbPred[2]
    assert bbPred[1] < bbPred[3]
    assert bbAnno[0] < bbAnno[2]
    assert bbAnno[1] < bbAnno[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bbPred[0], bbAnno[0])
    y_top = max(bbPred[1], bbAnno[1])
    x_right = min(bbPred[2], bbAnno[2])
    y_bottom = min(bbPred[3], bbAnno[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bbPred[2] - bbPred[0]) * (bbPred[3] - bbPred[1])
    bb2_area = (bbAnno[2] - bbAnno[0]) * (bbAnno[3] - bbAnno[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
def imgEval(pred,gt,finalHolder):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    # print(pred.shape,gt.shape)
    # print(pred,gt)
    _pred = pred.copy()
    _gt = gt.copy()

    # print(_pred[0])
    # _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    # _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    # i=0
    # while(i<_pred.shape[0]):
    #     print(_pred[i])
    #     _pred[i][2]+=_pred[i][0]
    #     _pred[i][3]+=_pred[i][1]
    #     i+=1
        
    # # print(_pred[0])
    # input()

    # _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    # _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
    boxes=_pred.copy()
    gt_boxes=_gt.copy()


    if boxes.shape[0] > 0:
        # _gt_overlaps = overlaps.max(axis=0)
        # print("the shapes",_gt_overlaps.shape,boxes.shape)
        #print('max_overlaps', _gt_overlaps, file=sys.stderr)
        # for j in range(len(_gt_overlaps)):
        #     if _gt_overlaps[j] > 0.5:
        #         continue
        #     #print(j, 'failed', gt_boxes[j],  'max_overlap:', _gt_overlaps[j], file=sys.stderr)
        j=0
        while(j<boxes.shape[0]):
            k=0
            box=boxes[j]
            fbox=[box[0],box[1],box[0]+box[2],box[3]+box[1]]
            print("THIS IS THE FBOX)",fbox)
            # fbox=box.copy().astype(np.float)
            maxi=0

            while(k<gt_boxes.shape[0]):
                gt_box = gt_boxes[k]
                gt_ibox = [gt_box[0],gt_box[1],gt_box[0]+gt_box[2],gt_box[3]+gt_box[1]]
                tmp=get_iou(fbox,gt_ibox)
                if(tmp>maxi):
                    maxi=tmp
                k+=1
            finalHolder.append([maxi,box[4]])                    


            j+=1
def imgEval2(pred,gt,iou_thresh,finalHolder):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    print(pred.shape,gt.shape)
    _pred = pred.copy()
    
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    print(overlaps.shape)
    overlaps=overlaps.T
    _p=pred.copy()

    #addin another parameter
    z=np.array([[0] for i in range(_p.shape[0])])
    print("shape of z is ")
    print(z.shape)
    _p=np.concatenate((_p,z),axis=1)
    #doing it for second time 
    _p=np.concatenate((_p,z),axis=1)
    print(_p)
    # input()

    for h in range(_gt.shape[0]):

        pred_overlap = overlaps[h]
        max_overlap, max_idx = pred_overlap.max(), pred_overlap.argmax()
        if(max_overlap>=_p[max_idx][6]):
            _p[max_idx][6]=max_overlap
            _p[max_idx][5]=1
        else:
            _p[max_idx][5]=1

        # finalHolder.append([max_overlap,_pred[h][4])
        # if max_overlap >= iou_thresh:
        #     if ignore[max_idx] == 0:
        #         recall_list[max_idx] = -1
        #         proposal_list[h] = -1
        #     elif recall_list[max_idx] == 0:
        #         recall_list[max_idx] = 1
     
        # r_keep_index = np.where(recall_list == 1)[0]
        # pred_recall[h] = len(r_keep_index)

    for h in range (_pred.shape[0]):

        finalHolder.append([_p[h][4],_p[h][5],_p[h][6]])
    


    # return pred_recall, proposal_list


def neel_image_eval(pred, gt,finalHolder):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    # print(pred.shape,gt.shape)
    print(pred,gt)
    input()
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
    # print(_gt[0])
    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    print(overlaps)
    # input()
    # print(overlaps.shape)
    print("----------")
    # input()

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

        finalHolder.append([max_overlap,_pred[h][4]])
        # if max_overlap >= iou_thresh:
        #     if ignore[max_idx] == 0:
        #         recall_list[max_idx] = -1
        #         proposal_list[h] = -1
        #     elif recall_list[max_idx] == 0:
        #         recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list




def evaluation(pred, gt_path, iou_thresh=0.5):
    a=open("/home/jatin/Desktop/Pytorch_Retinaface/data/widerface/val/label_NJIS.txt","r")
    finalholder=[]
    lines=a.readlines()
    gt_boxes=[]
    totalFacesAnno=0
    i=0
    j=0
    isFirst=True
    for line in lines:
        print("the line is"+line)
        line=line.strip()
        if(line.startswith("#")):
            j+=1
            if isFirst:
                isFirst=False
                prevName=line[1:].strip()
            else:
                #to besned for verification
                gt_boxesToSend=np.array(gt_boxes.copy())
                f=open("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/widerface_txt/"+prevName[:-4]+".txt","r")
                prevName=line[1:].strip()
                lines2=f.readlines()
                noPred=int(lines2[1])
                prebbox=[]
                if(noPred>0):
                    for line2 in lines2[2:]:
                        line2=line2.split()
                        labelpred=[float(x) for x in line2]
                        prebbox.append(labelpred)
                        
                    prebbox=np.array(prebbox)
                if(noPred>0 and gt_boxesToSend.shape[0]>0):
                
                    imgEval2(prebbox,gt_boxesToSend,iou_thresh,finalholder)
                f.close()
                # print(noPred)
                totalFacesAnno+=gt_boxesToSend.shape[0]

                #finding the prediictoin data boxed
                gt_boxes=[]
                # print(name)
                i+=1


        else:
            line=line.split(" ")
            label=[float(x) for x in line]
            # print("printing the labels")
            # print(label)
            gt_boxes.append(label[:4])

    # print(i,j,totalFacesAnno)
    a.close()

    # print(np.array(finalholder).shape)

    #read current model name
    a=open("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/lastModelRun.txt","r")
    model=a.read()
    a.close()

    
    import pickle
    if not os.path.exists("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/"):
        os.makedirs("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/")
    with open("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/finalHolderResults-"+model,'wb') as pickleFile:
    
    # Step 3
        pickle.dump(finalholder, pickleFile)

    a=open("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/totalFaces.txt","w")


    a.write(str(totalFacesAnno))
    print(finalholder)
    a.close()
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="./widerface_txt/")
    parser.add_argument('-g', '--gt', default='./ground_truth/')

    args = parser.parse_args()
    evaluation(args.pred, args.gt)












