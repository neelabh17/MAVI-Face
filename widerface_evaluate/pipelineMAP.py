"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
from os.path import join
import tqdm
import pickle
import argparse
import numpy as np
from IPython import embed
import sys
import pickle
import numpy as np
sys.path.append("..")
from widerface_evaluate.bbox import bbox_overlaps
from utils.evalResults import readData, reductionProcedures
import pickle
from data.compare_img import saveImages
from toolbox.makedir import make
from toolbox.pickleOpers import save,loadup
from toolbox.prManu import bestConf
from toolbox.plotter import prPlotter



parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='Resnet50_Final',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--savename', default='baseline', type=str, help='name of our save')
parser.add_argument('--dataset', default='val', type=str, help="on which dataset do we compare images")
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold_eval', default=0.03, type=float, help='confidence_threshold_eval')
parser.add_argument('--confidence_threshold_infer', default=0.055, type=float, help='confidence_threshold_infer')
parser.add_argument('--area_threshold', default=225, type=float, help='area threshold to remove small faces')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', default="False",type=str, help='show detection results')
parser.add_argument('--merge_images', default="False",type=str, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--mode', default='single', type=str, help='single: eval on single model, series: evaluate on training session comes with in built graph plotter')


args = parser.parse_args()



def neel_image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    # print("shapess are of pred and gt",pred.shape,gt.shape)
    # print(type(pred),pred)
    # print("--------------------------------",gt)
    pred=pred.astype(float)
    gt=gt.astype(float)
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

def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info

def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def getYesNoScoreList(pred,gt,ignore,iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    # print("shapess are of pred and gt",pred.shape,gt.shape)
    # print(type(pred),pred)
    # print("--------------------------------",gt)
    pred=pred.astype(float)
    gt=gt.astype(float)
    _pred = pred.copy()
    _gt = gt.copy()
    ynsList = np.zeros((_pred.shape[0],2))
    ynsList[...,1]=pred[...,4]
    # print("The confs list is \n",pred[...,4] )
    # print("The yns list is \n",ynsList )


    


    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)
   
    # print(overlaps.shape,"Overlapsssss\n",overlaps)
    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        # print("gt overlap\n",gt_overlap)
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        # print(max_overlap,max_idx)
        if max_overlap >= iou_thresh:
            # if ignore[max_idx] == 0:
            # my change made
            if False:
                ynsList[max_idx][0] = -1
            elif ynsList[h][0] == 0:
                ynsList[h][0] = 1

    return ynsList

def getYesNoScoreList2(pred,gt,ignore,iou_thresh,fileName):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    # print("shapess are of pred and gt",pred.shape,gt.shape)
    # print(type(pred),pred)
    # print("--------------------------------",gt)
    pred=pred.astype(float)
    gt=gt.astype(float)
    _pred = pred.copy()
    _gt = gt.copy()
    ynsList = np.zeros((_pred.shape[0],2))
    ynsList[...,1]=pred[...,4]
    # print("The confs list is \n",pred[...,4] )
    # print("The yns list is \n",ynsList )


    


    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)
   
    # print(overlaps.shape,"Overlapsssss\n",overlaps)
    overlaps=overlaps.T
    for h in range(_gt.shape[0]):
        # for igno in ignore:
        #     print(igno)
        # input()
        # print(fileName)
        # print(fileName+str(h))
        # print(fileName+str(h) in ignore)
        
        if (fileName+str(h)) in ignore:
            print("ignoring"+fileName+str(h))
        else:
            pred_overlap = overlaps[h]
            # print("gt overlap\n",pred_overlap)
            max_overlap, max_idx = pred_overlap.max(), pred_overlap.argmax()
            # print(max_overlap,max_idx)
            if max_overlap >= iou_thresh:
                # if ignore[max_idx] == 0:
                # my change made
                if False:
                    ynsList[max_idx][0] = -1
                elif ynsList[max_idx][0] == 0:
                    ynsList[max_idx][0] = 1

    return ynsList

def givePRCurve(pr_data_collector,count_face):
    # pr_data_collector= nx2 (yes/no,conf)
    #returns nx3 (prec,recall,conf)
    #---------------------------------------------
    # sort the data according to scores
    pr_data_collector.view("f8,f8").sort(order=["f1"],axis=0)
    pr_data_collector=pr_data_collector[::-1,...]
    # print(pr_data_collector.shape)

    # now looping over it 
    tp=0
    fp=0
    my_pr_curve=np.zeros((pr_data_collector.shape[0],3))
    
    for i, prPoint in enumerate(my_pr_curve):
        if(pr_data_collector[i][0]==1):
            tp+=1
        elif(pr_data_collector[i][0]==0):
            fp+=1
        else:
            input("An error occured")
        
        prPoint[0]=(tp/(tp+fp))# this is precision
        prPoint[1]=tp/count_face
        prPoint[2]=pr_data_collector[i][1]

    return my_pr_curve

def getGTIgnoreSet(category):

    file=open("/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/errorAnal/fn.pickle","rb")
    a=pickle.load(file)
    file.close()
    # errorList=["Blur","Small","Dark","Tree","occulusion","in group","dif object","non annotated face","less iou"]
    errorList=["Small","Dark","in group","not predicted","less iou"]



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
    return setList[category],len(setList[category])


def neelEvaluation(iou_thresh,modelPath,seriesData=None,writer=None):
    count_face = 0
    thresh_num = 1000
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    aps=[]
    my_aps=[]
    totalFacesAnno=0
    i=0
    j=0
    #load val dataset ground truth
    fileName=join(os.getcwd(),"data","widerface","val","label.pickle")
    gts=loadup(fileName)

    #load the predbbooxes dataset ground truth
    evalDataFolder=join(os.getcwd(),"evalData")
    if(seriesData is not None):
        modelEvalFolder = join(os.getcwd(),"evalData",seriesData["seriesName"]+f"_inferConf={args.confidence_threshold_infer}","outResults")
        fileName = join(modelEvalFolder,"outResults_{}_epoch_{}.pickle".format(args.dataset,seriesData["epoch"]))
    else:
        modelEvalFolder = join(os.getcwd(),"evalData",os.path.basename(modelPath).strip(".pth")+f"_inferConf={args.confidence_threshold_infer}","outResults")
        fileName = join(modelEvalFolder,"outResults_{}.pickle".format(args.dataset))


    preds=loadup(fileName)

    #my addition for pr implementation acccording to jonathan huis article
    pr_data_collector=np.array([]).reshape(0,2)
    # ignore,facesRemoved=getGTIgnoreSet("Small")
    ignore=[]
    for i,fileName in enumerate(gts):
        # print(i,fileName)

        #because ppickle file doesnt load files in form of numpy stuff
        gt_boxesToSend=np.array(gts[fileName])
        gt_boxesToSend=gt_boxesToSend[...,:4]
        gt_boxesToSend=gt_boxesToSend.astype(float)

        pred_data=preds[fileName]
        dets,predbox=reductionProcedures(pred_data,args.nms_threshold,args.confidence_threshold_infer)
        #removing small faces
        area_thresh=args.area_threshold   #in pixels
        predbox=predbox[np.where(np.multiply(predbox[:,2],predbox[:,3])>=area_thresh)[0]]
        # print("---------------")
        # print(np.multiply(gt_boxesToSend[:,2],gt_boxesToSend[:,3]))
        # print(gt_boxesToSend)
        # print(gt_boxesToSend.shape)
        gt_boxesToSend=gt_boxesToSend[np.where(np.multiply(gt_boxesToSend[:,2],gt_boxesToSend[:,3])>=area_thresh)[0]]
        # print(gt_boxesToSend.shape)
        if(predbox.shape[0]>0 and gt_boxesToSend.shape[0]>0):
            # ignore = np.zeros(gt_boxesToSend.shape[0])

            count_face+=len(gt_boxesToSend)
            pred_recall, proposal_list = neel_image_eval(predbox, gt_boxesToSend, ignore, iou_thresh)
            _img_pr_info = img_pr_info(thresh_num, predbox, proposal_list, pred_recall)
            pr_curve += _img_pr_info

            #my addition for pr implementation acccording to jonathan huis article
            yns_List=getYesNoScoreList2(predbox,gt_boxesToSend,ignore,iou_thresh,fileName)
            pr_data_collector=np.concatenate((pr_data_collector,yns_List),axis=0)
        
        #i am thinking of adding the other cases as well when no gt boxes and whn no pred boxes ok well do that in next version


            
    
    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

    #my addition for pr implementation acccording to jonathan huis article
    # print("pr curve",pr_data_collector,np.array(pr_data_collector))
    my_pr_curve=givePRCurve(pr_data_collector,count_face)

    propose = my_pr_curve[:, 0]
    recall = my_pr_curve[:, 1]
    for i in range(len(recall)):
        if(seriesData is not None):
            # writer.add_scalars("prCurve",{f'epoch={seriesData["epoch"]} iou_thresh={iou_thresh}':propose[i]},recall[i]*1000)
            writer.add_scalars(f"prCurve/iou_thresh={iou_thresh}",{f'epoch={seriesData["epoch"]} ':propose[i]},recall[i]*1000)
            pass
        else:
            writer.add_scalars(f"prCurve/iou_thresh={iou_thresh}",{'epoch=1':propose[i]},recall[i]*1000)
            pass

    writer.flush()
    my_ap=voc_ap(recall,propose)
    print("my ap is coming out to be",my_ap)
    if(seriesData is not None):
        prFolder=join(os.path.dirname(modelEvalFolder),"prData")
        make(prFolder)
        prFileName = join(prFolder,"prCurve{}_{}_epoch_{}.pickle".format(int(iou_thresh*100),args.dataset,seriesData["epoch"]))
    else:
        prFolder=join(os.path.dirname(modelEvalFolder),"prData")
        make(prFolder)
        prFileName = join(prFolder,"prCurve{}_{}.pickle".format(int(iou_thresh*100),args.dataset))

    a=open(prFileName,"wb")
    pickle.dump(my_pr_curve,a)
    a.close()
    # saving manual plots
    # prPlotter(prFileName)


    print("saving pickle for optimise")
    if(iou_thresh==0.3):
        mr,mp,mf=bestConf(prFileName)
        if(seriesData is not None):
            writer.add_text("Max Recall",mr,seriesData["epoch"])
            writer.add_text("Max Precision",mp,seriesData["epoch"])
            writer.add_text("Max F1 Score",mf,seriesData["epoch"])
        else:
            writer.add_text("Max Recall",mr,1)
            writer.add_text("Max Precision",mp,1)
            writer.add_text("Max F1 Score",mf,1)
        # writer.flush()
    my_aps.append(my_ap)
    #correctnig the nan values that may have arrived due to division by zero
    for xe in pr_curve:
        if(np.isnan(xe[0])):
            xe[0]=1
    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    # print(recall)

    ap = voc_ap(recall, propose)
    aps.append(ap)

    # evalDataFolder="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/evalData/"
    # a=open(evalDataFolder+args.trained_model.strip(".pth").strip("/weights/")+"/pr{}.pickle".format(int(iou_thresh*100)),"wb")
    # pickle.dump(pr_curve,a)
    # a.close()

    
    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    # print("Medium Val AP: {}".format(aps[1]))
    # print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")

    return aps[0]

def MAPCalcAfterEval(newargs=args ,modelPath=None,seriesData=None,writer=None):
    print(f'The infer confidence is ={args.confidence_threshold_infer}')
    if(seriesData is None):
        assert(newargs.mode=="single")
    
    if(seriesData is not None):
        evalModelFolder=join(os.getcwd(),"evalData",seriesData["seriesName"]+f"_inferConf={newargs.confidence_threshold_infer}")
    else:
        evalModelFolder=join(os.getcwd(),"evalData",os.path.basename(modelPath).strip(".pth")+f"_inferConf={newargs.confidence_threshold_infer}")

    if(newargs.save_image=="True"):
        model_name=os.path.basename(modelPath).strip(".pth")
        saveImages(model_name,newargs.nms_threshold,newargs.confidence_threshold_infer,newargs.dataset,newargs.savename,newargs.area_threshold,newargs.merge_images)
        # n=int(input("Want to continue?"))
        # if(n==0):
        #     exit()
        exit()


    # n=int(input("0 for NJIS,,,, 1 for Widerface:  "))
    #doing this intentionally so that i dint have to inupt it every single time
    n=0

    ious=[0.25+int(0.05*i*100)/100 for i in range(15)]

    iouVsAP=[]
    for i in ious:
        iouVsAP.append([i,neelEvaluation(i,modelPath,seriesData,writer=writer)])
    mAP_025=0
    mAP_05=0

    mapFolder=join(evalModelFolder,"mapData","results")
    make(mapFolder)


    if(seriesData is not None):
        a=open(join(mapFolder,"results_epoch_{}.txt".format(seriesData["epoch"])),"w")
    else:
        a=open(join(mapFolder,"results.txt"),"w")

    for itemer in iouVsAP:
        if( itemer[0]>=0.5):
            mAP_05+=itemer[1]
        print(itemer[0], "\t AP={:.4f}".format(itemer[1]))
        a.write(str(itemer[0])+"\t AP={:.4f}".format(itemer[1]))
        a.write("\n")
        mAP_025+=itemer[1]

    print("=================================================")
    print("mAP 0.25:0.05:0.95 is  : {:.4f}".format(mAP_025/len(ious)))
    print("mAP 0.5:0.05:0.95 is  : {:.4f}".format(mAP_05/10))
    a.write("===============================================\nmAP 0.25:0.05:0.95 is :{:.4f} ".format(mAP_025/len(ious)))
    a.write("\nmAP 0.5:0.05:0.95 is : {:.4f}".format(mAP_05/10))
    a.close()

    # save the same in a pickle file not really helpful but still let it be there
    if seriesData is not None:
        save(iouVsAP,join(mapFolder,"results_epoch_{}.pickle".format(seriesData["epoch"])))
    else:
        save(iouVsAP,join(mapFolder,"results.pickle"))

    return mAP_05/10

if __name__ == '__main__':

    if(args.save_image=="True"):
        model_name=args.trained_model.strip(".pth").strip("/weights/")
        saveImages(model_name,args.nms_threshold,args.confidence_threshold_infer,args.dataset,args.savename,args.area_threshold,args.merge_images)
        # n=int(input("Want to continue?"))
        # if(n==0):
        #     exit()
        exit()


    # n=int(input("0 for NJIS,,,, 1 for Widerface:  "))
    #doing this intentionally so that i dint have to inupt it every single time
    n=0

    ious=[0.25+int(0.05*i*100)/100 for i in range(15)]
    # ious=[0.5+int(0.05*i*100)/100 for i in range(10)]
    iouVsAP=[]
    for i in ious:
        # print(i)
        # input()
        iouVsAP.append([i,neelEvaluation(i,n)])
    mAP_025=0
    mAP_05=0
    evalDataFolder="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/evalData/"
    a=open(evalDataFolder+args.trained_model.strip(".pth").strip("/weights/")+"/resultspickle.txt","w")
    # a=open(evalDataFolder+args.trained_model.strip(".pth").strip("/weights/")+"/resultspicklesmall.txt","w")
    for itemer in iouVsAP:
        if( itemer[0]>=0.5):
            mAP_05+=itemer[1]
        print(itemer[0], "\t AP={:.4f}".format(itemer[1]))
        a.write(str(itemer[0])+"\t AP={:.4f}".format(itemer[1]))
        a.write("\n")
        mAP_025+=itemer[1]

    print("=================================================")
    print("mAP 0.25:0.05:0.95 is  : {:.4f}".format(mAP_025/len(ious)))
    print("mAP 0.5:0.05:0.95 is  : {:.4f}".format(mAP_05/10))
    a.write("===============================================\nmAP 0.25:0.05:0.95 is :{:.4f} ".format(mAP_025/len(ious)))
    a.write("\nmAP 0.5:0.05:0.95 is : {:.4f}".format(mAP_05/10))
    a.close()

    a=open(evalDataFolder+args.trained_model.strip(".pth").strip("/weights/")+"/results.pickle","wb")
    # a=open(evalDataFolder+args.trained_model.strip(".pth").strip("/weights/")+"/resultssmall.pickle","wb")
    import pickle
    pickle.dump(iouVsAP,a)
    a.close()
    

    # print(neelEvaluation(args.pred, args.gt,0.3))












