import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch

gt_folder="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/widerface_evaluate/ground_truth/images/"
pred_folder="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/widerface_evaluate/widerface_txt/Resnet50_epoch_20_noGrad/imgResults/"
plt.rcParams["figure.figsize"] = (15,10)
images=[]
for i in range(10):
    print(i)
    gt_img=gt_folder+str(i)+".jpg"
    pred_img=pred_folder+str(i)+".jpg"

    # ok=(cv2.imread(gt_img,cv2.IMREAD_COLOR))
    # print(ok[...,::-1].shape)
    gt=cv2.imread(gt_img,cv2.IMREAD_COLOR)
    gt=cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
    gt=torch.tensor(gt).permute(2,0,1)
    pred=cv2.imread(pred_img,cv2.IMREAD_COLOR)
    pred=cv2.cvtColor(pred,cv2.COLOR_BGR2RGB)
    pred=torch.tensor(pred).permute(2,0,1)
    # pred=torch.tensor(cv2.imread(pred_img,cv2.IMREAD_COLOR)).permute(2,0,1)
    # gt=gt.unsqueeze(0)
    # pred=pred.unsqueeze(0)
    # print(gt.shape)
    # print(pred.shape)
    
    images.append(gt)
    images.append(pred)
    # images=torch.cat((images,gt),0)
    # print(images.shape)
    # images.append(pred)
images=torch.stack(images)
print(images.shape)
grid=torchvision.utils.make_grid(images,nrow=2)
plt.figure(figsize=(30,30))
plt.imshow(np.transpose(grid,(1,2,0)))
print(np.transpose(grid,(1,2,0)).shape)
plt.show()
# print("labels ",labels)





    #concatanate image Horizontally
    # img_concate_Hori=np.concatenate((gt,pred),axis=1)
    # # cv2.imshow(str(i),img_concate_Hori)
    # plt.imshow(img_concate_Hori)
    # plt.show()