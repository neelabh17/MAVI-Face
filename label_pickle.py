
file=open("/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/data/widerface/train/label.txt","r")
lines=file.readlines()
file.close()
# ill like to do this my way now reallyyou should seee how i do this
i=0

ll={}
lenOfFile=len(lines)
while(i<lenOfFile):
    line=lines[i]
    fn=""
    if(line.startswith("#")):
        fn=line[1:].strip()
        ll[fn]=[]
    j=i+1
    notFound=True
    while(j<lenOfFile and notFound):
        line=lines[j]
        if(line.startswith("#")):
            notFound=False
        else:
            #since the value are given in float we just can use the float value directly to convert themm to integer
            annoList=list(map(lambda x:int(float(x)),line.strip("\n").split()))
            ll[fn].append(annoList)
            j+=1
    i=j

import pickle
filename=open("/content/drive/My Drive/RetinaFace/Pytorch_Retinaface/data/widerface/train/label.pickle","wb")
pickle.dump(ll,filename)
filename.close()