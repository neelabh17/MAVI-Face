import matplotlib.pyplot as plt 
import pickle
import os
def lossGraphPlotter(fileName,viewMode=False,saveMode=True):
    '''
    # fileName: full file path
    '''
    # loading the pickle file
    f=open(fileName,"rb")
    lossCollec=pickle.load(f)
    f.close()

    
    ep=[]
    vl=[]
    tl=[]
    ol=[]
    for ls in lossCollec:
        ep.append(ls["epoch"])
        # tl.append(ls["TrainLoss"]/1231)
        vl.append(ls["valLoss"])
        tl.append(ls["trainLoss"])
        ol.append(ls["ohemLoss"])


    # print(len(ep),len(vl))
    minVal=min(vl)
    minvl=[minVal]*len(ep)
    plt.clf()
    plt.rcParams["figure.figsize"] = (15,10)
    # for x in a:
    #     # plotting the points

    
    plt.plot(ep,vl,"g")
    plt.plot(ep,tl,"r")
    plt.plot(ep,ol,"orange")
    plt.plot(ep,minvl,"b",linestyle="dashed",marker="o",markevery=[ep[vl.index(min(vl))]-1])
    # i=0 
    # while(i < len(ep)):
    #     print(ep[i],vl[i])
    #     i+=1


    # naming the x axis 
    plt.xlabel('Epoch') 
    # naming the y axis 
        
    # giving a title to my graph 
    title=os.path.basename(os.path.dirname(os.path.dirname(fileName) ) )
    # print(fileName.strip(".pickle").split("/")[-3])

    plt.ylabel("LOSS per image") 
    plt.title(title+":  Loss(per image) Vs Epoch ") 
    plt.text(ep[len(ep)-1]*.10,max(vl[len(ep)-1],tl[len(ep)-1],ol[len(ep)-1])+(-max(vl[len(ep)-1],tl[len(ep)-1],ol[len(ep)-1])+max(vl[0],tl[0],ol[0]))*0.5,"Validation Loss per image (min={0:.2f} at epoch {1})".format(min(vl),ep[vl.index(min(vl))]),fontsize=12,color="green")
    plt.text(ep[len(ep)-1]*.10,max(vl[len(ep)-1],tl[len(ep)-1],ol[len(ep)-1])+(-max(vl[len(ep)-1],tl[len(ep)-1],ol[len(ep)-1])+max(vl[0],tl[0],ol[0]))*0.6,"Training Loss per image (min={0:.2f} at epoch {1})".format(min(tl),ep[tl.index(min(tl))]),fontsize=12,color="red")
    plt.text(ep[len(ep)-1]*.10,max(vl[len(ep)-1],tl[len(ep)-1],ol[len(ep)-1])+(-max(vl[len(ep)-1],tl[len(ep)-1],ol[len(ep)-1])+max(vl[0],tl[0],ol[0]))*0.7,"Ohem Loss per image (min={0:.2f} at epoch {1})".format(min(ol),ep[ol.index(min(ol))]),fontsize=12,color="orange")
    if(saveMode):
        plt.savefig(fileName.strip(".pickle")+"-graph.png")
    # plt.savefig( fileName.strip(".pickle")+"-Loss_per_image.jpg")
   
       
    
    # function to show the plot 
    if(viewMode):
        plt.show() 

def mapGraphPlotter(fileName,viewMode=False,saveMode=True):
    '''
    # fileName: full file path
    '''
    # loading the pickle file
    f=open(fileName,"rb")
    lossCollec=pickle.load(f)
    f.close()

    
    ep=[]
    MAP=[]
    
    for ls in lossCollec:
        ep.append(ls["epoch"])
        # tl.append(ls["TrainLoss"]/1231)
        MAP.append(ls["map"])
        


    # print(len(ep),len(vl))
    #maximum map
    maxVal=max(MAP)
    maxVal=[maxVal]*len(ep)

    # reference map
    refVal=0.3631
    refVal=[refVal]*len(ep)
    
    plt.clf()

    plt.rcParams["figure.figsize"] = (15,10)
    

    
    plt.plot(ep,MAP,"g")
    plt.plot(ep,maxVal,"b",linestyle="dashed",marker="o",markevery=[ep[vl.index(min(vl))]-1])
    plt.plot(ep,refVal,"r",linestyle="dashed")
    


    # naming the x axis 
    plt.xlabel('Epoch') 
    # naming the y axis 
        
    # giving a title to my graph 
    title=os.path.basename(os.path.dirname(os.path.dirname(fileName) ) )

    plt.ylabel("mAP") 
    plt.title(title+":  mAP (0.5:0.05:0.95) ") 
    plt.text(ep[len(ep)-1]*.50,(MAP[0]+MAP[len(ep)-1]-MAP[0])*0.5,"mAP (max={0:.4f} at epoch {1})".format(max(MAP),ep[MAP.index(max(MAP))]),fontsize=12,color="green")
    plt.text(ep[len(ep)-1]*.10,(MAP[0]+MAP[len(ep)-1]-MAP[0])*0.6,"Max mAP achieved = {0:.4f}".format(max(MAP)),fontsize=12,color="blue")
    plt.text(ep[len(ep)-1]*.10,(MAP[0]+MAP[len(ep)-1]-MAP[0])*0.7,"Pretrained reference mAP n={0:.4f}".format(min(refVal)),fontsize=12,color="red")
    if(saveMode):
        plt.savefig(fileName.strip(".pickle")+"-graph.png")
    # plt.savefig( fileName.strip(".pickle")+"-Loss_per_image.jpg")
   
       
    
    # function to show the plot 
    if(viewMode):
        plt.show() 
