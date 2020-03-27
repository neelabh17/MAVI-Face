# Read from pickle file
import pickle
a=open("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/lastModelRun.txt","r")
model=a.read()
a.close()
# Step 2
with open("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/finalHolderResults-"+model, 'rb') as config_dictionary_file:
 
    # Step 3
    finalHolder = pickle.load(config_dictionary_file)
#read total faces
a=open("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/totalFaces.txt","r")
totaFaces=int(a.read())
# print(totaFaces)
a.close()


#reverse sorting
finalHolder.sort(key=lambda x:x[1],reverse=True)


# put precision and recall into two different arrays
a={"0.4":[],"0.5":[],"0.55":[],"0.6":[],"0.65":[],"0.7":[],"0.75":[],"0.8":[],"0.85":[],"0.9":[],"0.95":[]}
precision={"0.4":[],"0.5":[],"0.55":[],"0.6":[],"0.65":[],"0.7":[],"0.75":[],"0.8":[],"0.85":[],"0.9":[],"0.95":[]}
recall={"0.4":[],"0.5":[],"0.55":[],"0.6":[],"0.65":[],"0.7":[],"0.75":[],"0.8":[],"0.85":[],"0.9":[],"0.95":[]}
totalFaces =575
for x in a:

    i=0
    currentCorrect=0

    while(i<len(finalHolder)):
        if(finalHolder[i][0]>float(x)):
            adder=1
        else:
            adder=0
        currentCorrect+=adder
        precision[x].append(currentCorrect/(i+1))
        recall[x].append(currentCorrect/totalFaces)


        i+=1

# precision.append(0)
# recall.append(1)



# print out the specifics of the model
stepper={}
area={}
f=open("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/resuls.txt","w")
for x in a:
    l=len(precision[x])
    stepper[x]=[0 for i in range (l)]
    maxi=0
    posi=1
    area[x]=0 
    i=0
    
    width = recall[x][l-1]
    while(i<l):
        
        if(precision[x][l-1-i]>=maxi):
            area[x] +=(width - recall[x][l-1-i]) * maxi
            width = recall[x][l-1-i]
            maxi=precision[x][l-1-i]
            # area+=((posi-recall[x][l-1-i])*maxi)
            posi=recall[x][l-1-i]

        stepper[x][l-1-i]=maxi

        i+=1

    f.write(str(x)+"\t---------->>>>>"+str(area[x])+"\n")
    print(x,"\t---------->>>>>",area[x])



#plotting the graphs
import os
if not os.path.exists("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/graphs/"):
    os.makedirs("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/graphs/")
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = (15,10)
for x in a:
    # plotting the points  
    plt.plot(recall[x], precision[x],"r") #x,y
    plt.plot(recall[x],stepper[x],"g")
    
    # naming the x axis 
    plt.xlabel('Recall') 
    # naming the y axis 
    plt.ylabel('Precision') 
    
    # giving a title to my graph 
    plt.title("IOU at "+ x+", with AP= "+(str)(area[x])) 
    plt.savefig("/home/jatin/Desktop/Pytorch_Retinaface/widerface_evaluate/"+model+"/graphs/IOU:"+ x+".jpg")
    # function to show the plot 
    plt.show() 
