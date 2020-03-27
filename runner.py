import os

i=4
while(i<=20):
    os.system("python test_widerface.py --trained_model ./weights/Resnet50_epoch_{}_lr_div12.pth".format(i))
    # os.system("cd widerface_evaluate")
    os.system("python ./widerface_evaluate/neelEvaluate.py ")
    os.system("python ./widerface_evaluate/graphProcucer.py")
    # os.system( "cd ..")
    

    i+=4


# os.system("python test_widerface.py --trained_model ./weights/Resnet50_Final-ly.pth".format(i))
# # os.system("cd widerface_evaluate")
# os.system("python ./widerface_evaluate/neelEvaluate.py ")
# os.system("python ./widerface_evaluate/graphProcucer.py")
# # os.system( "cd ..")


i=4
while(i<=28):
    a=open("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/Resnet50_epoch_"+str(i)+"/resuls.txt".format(i),"w")
    lines=a.readlines()
    print(lines[1])

    i+=1