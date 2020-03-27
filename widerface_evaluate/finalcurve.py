
i=4
epoch=[]
ap=[]
while(i<=28):
    print(i)
    a=open("Resnet50_epoch_"+str(i)+"/resuls.txt","r")
    lines=a.readlines()
    j=0
    for line in lines:
        j+=1
        line=line.strip()
        line=line.split(">")
        if(j==2):
            print(line[len(line)-1])
            ap.append(float(line[len(line)-1]))
            epoch.append(i)
        
    a.close()

    i+=4
print(epoch)
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = (10,2)

plt.plot(epoch, ap,"r") #x,y


# naming the x axis 
plt.xlabel('Epochs') 
# naming the y axis 
plt.ylabel('AP') 

# giving a title to my graph 
plt.title("At IoU 0.5") 
plt.savefig("AP vs Epoch :0.5.jpg")
# function to show the plot 
plt.show()