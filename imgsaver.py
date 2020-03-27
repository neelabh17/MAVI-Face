import os

i=8
while(i<=40):
    if(i==40):
        os.system("python test_widerface.py --trained_model \'./weights/Resnet50_Finally_FT_LRe5.pth\'".format(i))
        ok=1
    else:

        os.system("python test_widerface.py --trained_model \'./weights/Resnet50_epoch_{}_noGrad_FT_LRe5.pth\'".format(i))
    i+=2