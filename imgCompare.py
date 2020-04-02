import cv2
import numpy as np
pathBB="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/widerface_evaluate/widerface_txt/Resnet50_epoch_28_noGrad_FT_Adam_lre3/imgResults"
pathGT="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/widerface_evaluate/ground_truth/images"

pathToSave="/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/errorAnal/images_for_comparision"

i=0
while(i<351):
    a=cv2.imread(pathBB+"/{}.jpg".format(i))
    b=cv2.imread(pathGT+"/{}.jpg".format(i))
    c=np.concatenate((a,b),axis=0)
    image=c
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    org = (50, 50) 
    org2 = (50, 50+c.shape[0]//2) 
    
    # fontScale 
    fontScale = 1
    
    # Blue color in BGR 
    color = (255, 0, 0) 
    color2 = (0, 0, 255) 
    
    # Line thickness of 2 px 
    thickness = 2
    c=cv2.putText(image, 'Model\'s Prediction', org, font, fontScale, color, thickness, cv2.LINE_AA) 
    c=cv2.putText(image, 'Ground Truth', org2, font, fontScale, color, thickness, cv2.LINE_AA) 
    cv2.imwrite(pathToSave+"/{}.jpg".format(i),c)
    print("{}-th image done saving".format(i))

    i+=1