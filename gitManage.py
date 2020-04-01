import os


filer=open("gitToken.txt")
f=filer.readline()
gitToken=f.strip()
filer.close()
print("Current gitToken is :{}\n".format(gitToken))
a=int(input("1: New token\n2: Same old one\n"))
if(a==1):
    gitToken=input("Enter new Token")
    filer=open("gitToken.txt","w")
    filer.write(gitToken)
    filer.close()
else:
    filer=open("gitToken.txt")
    f=filer.readline()
    gitToken=f.strip()
    filer.close()



def spacer():
    print("##################################################")
print("Welcome to the Github Updater Program\nPlease Select:\n\n1: Make temp folder with MAVI-Face Mounted\n2: Delete temp folder\n3: Upload changes to Github\n4: Get Changes from Github")
n=int(input())
if(n==1): #for Make temp folder with MAVI-Face Mounted
    print("Changing Directory to cd /content/drive/My Drive/RetinaFace")
    spacer()
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("Checking if temp already exists")
    spacer()
    if not os.path.exists("temp"):
        print("Temp Doesn't exist creating one and mounting it with the MAVI-Face.git")
        spacer()
        
        os.system("mkdir temp")
        print("Directory Made")
        spacer()
        
        os.system("! git clone \'https://"+gitToken+"@github.com/neelabh17/MAVI-Face.git\' temp")
        print("MAVI-Face.git Cloned")
    else:
        print("The temp folder already exists")

elif(n==2):#Delete temp folder
    print("Changing Directory to cd /content/drive/My Drive/RetinaFace")
    spacer()
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("Checking if temp already exists")
    spacer()
    if not os.path.exists("temp"):
        print("Temp doesn't exist already. so cant delete it")
        
    else:
        print("The temp folder already exists\nDeleting it")
        spacer()
        
        os.system("rm -rf temp")
        print("Directory Deleted")
elif(n==3):#Upload changes to Github
    print("Changing Directory to cd /content/drive/My Drive/RetinaFace")
    spacer()
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("Checking if temp already exists")
    spacer()
    if not os.path.exists("temp"):
        print("Temp doesn't exist")
        print("Lets put it up there")
        print("Temp Doesn't exist creating one and mounting it with the MAVI-Face.git")
        spacer()
        
        os.system("mkdir temp")
        print("Directory Made")
        spacer()
        
        os.system("! git clone \'https://"+gitToken+"@github.com/neelabh17/MAVI-Face.git\' temp")
        print("MAVI-Face.git Cloned")

        print("The temp folder has been set up")
        print("Rippling changes to temp folder")
        spacer()
        
        os.system("rsync -aP --exclude \'/errorAnal/images_for_comparision/*.jpg\' --exclude=\'/weights/*.pth\' --exclude=\'.git\' --exclude=\'widerface_evaluate/**images/**.jpg\' --exclude=\'widerface_evaluate/**imgResults/**.jpg\' --exclude=\'widerface_evaluate/**.txt\' --exclude=\'data/**.jpg\'  \'./Pytorch_Retinaface/\' \'./temp/\'")
        print("Changing directory into temp")
        spacer()
        
        os.chdir("temp")
        print("Entering Github credentials")
        spacer()
        
        os.system("git config --global user.email \'neelabh.madan@outlook.com\'")
        spacer()
        
        os.system("git config --global user.name \'neelabh17\'")
        print("Credentials Entered")
        print("Staging(Add) files")
        spacer()
        
        os.system("git add .")
        print("Commiting files")
        commit_name=input("Enter commit Name")
        spacer()
        
        os.system("git commit -m\'"+commit_name+"\'")
        print("Pushing it onto Github")
        spacer()
        
        os.system("git push origin master")
        print("\nIt has been pushed")


        
    else:
        print("The temp folder already exists")

        print("The temp folder already exists\nDeleting it")
        spacer()
        
        os.system("rm -rf temp")

        print("Temp Doesn't exist creating one and mounting it with the MAVI-Face.git")
        spacer()
        
        os.system("mkdir temp")
        print("Directory Made")
        spacer()
        
        os.system("! git clone \'https://"+gitToken+"@github.com/neelabh17/MAVI-Face.git\' temp")
        print("MAVI-Face.git Cloned")

        print("The temp folder has been set up")

        print("Rippling changes to temp folder")
        spacer()
        
        os.system("rsync -aP --exclude \'/errorAnal/images_for_comparision/*.jpg\' --exclude=\'/weights/*.pth\' --exclude=\'.git\' --exclude=\'widerface_evaluate/**images/**.jpg\' --exclude=\'widerface_evaluate/**imgResults/**.jpg\' --exclude=\'widerface_evaluate/**.txt\' --exclude=\'data/**.jpg\'  \'./Pytorch_Retinaface/\' \'./temp/\'")
        print("Changing directory into temp")
        spacer()
        
        os.chdir("temp")
        print("Entering Github credentials")
        spacer()
        
        os.system("git config --global user.email \'neelabh.madan@outlook.com\'")
        spacer()
        
        os.system("git config --global user.name \'neelabh17\'")
        print("Credentials Entered")
        print("Staging(Add) files")
        spacer()
        
        os.system("git add .")
        print("Commiting files")
        commit_name=input("Enter commit Name")
        spacer()
        
        os.system("git commit -m\'"+commit_name+"\'")
        print("Pushing it onto Github")
        spacer()
        
        os.system("git push origin master")
        print("\nIt has been pushed")

elif(n==4):#Get Changes from Github
    print("Changing Directory to cd /content/drive/My Drive/RetinaFace")
    spacer()
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("Checking if temp already exists")
    spacer()
    if not os.path.exists("temp"):
        print("Temp doesn't exist already. lets create it")
        print("Lets put it up there")
        print("Temp Doesn't exist creating one and mounting it with the MAVI-Face.git")
        spacer()
        
        os.system("mkdir temp")
        print("Directory Made")
        spacer()
        
        os.system("! git clone \'https://"+gitToken+"@github.com/neelabh17/MAVI-Face.git\' temp")
        print("MAVI-Face.git Cloned")


    else:
        print("The temp folder already exists")
        print("Changing directory into temp")
        spacer()
        
        os.chdir("temp")
        print("Entering Github credentials")
        spacer()
        
        os.system("git config --global user.email \'neelabh.madan@outlook.com\'")
        spacer()
        
        os.system("git config --global user.name \'neelabh17\'")
        
        print("Lets get a pull request")
        spacer()
        
        os.system("git pull origin master")

    print("Changing directory back")
    spacer()
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("temp folder is upto date. Lets update the Pytorch Retinaface folder")
    spacer()
    
    os.system("rsync -aP --exclude=\'.git\' \'./temp/\' \'./Pytorch_Retinaface/\'")



else:
    print("You Havent Selected the right option. Bye!!!")
print("Changing directory back to Retinaface pytorch")
spacer()

os.chdir("/content/drive/My Drive/RetinaFace/Pytorch_Retinaface")