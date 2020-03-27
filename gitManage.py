import os

print("Welcome to the Github Updater Program\nPlease Select:\n\n1: Make temp folder with MAVI-Face Mounted\n2: Delete temp folder\n3: Upload changes to Github\n4: Get Changes from Github")
n=int(input())
if(n==1): #for Make temp folder with MAVI-Face Mounted
    print("Changing Directory to cd /content/drive/My Drive/RetinaFace")
    print("#################################################")
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("Checking if temp already exists")
    print("#################################################")
    if not os.path.exists("temp"):
        print("Temp Doesn't exist creating one and mounting it with the MAVI-Face.git")
        print("#################################################")
        
        os.system("mkdir temp")
        print("Directory Made")
        print("#################################################")
        
        os.system("! git clone \'https://2965dbeaddd4f53bf03281027f453c69db771392@github.com/neelabh17/MAVI-Face.git\' temp")
        print("MAVI-Face.git Cloned")
    else:
        print("The temp folder already exists")

elif(n==2):#Delete temp folder
    print("Changing Directory to cd /content/drive/My Drive/RetinaFace")
    print("#################################################")
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("Checking if temp already exists")
    print("#################################################")
    if not os.path.exists("temp"):
        print("Temp doesn't exist already. so cant delete it")
        
    else:
        print("The temp folder already exists\nDeleting it")
        print("#################################################")
        
        os.system("rm -rf temp")
        print("Directory Deleted")
elif(n==3):#Upload changes to Github
    print("Changing Directory to cd /content/drive/My Drive/RetinaFace")
    print("#################################################")
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("Checking if temp already exists")
    print("#################################################")
    if not os.path.exists("temp"):
        print("Temp doesn't exist")
        print("Lets put it up there")
        print("Temp Doesn't exist creating one and mounting it with the MAVI-Face.git")
        print("#################################################")
        
        os.system("mkdir temp")
        print("Directory Made")
        print("#################################################")
        
        os.system("! git clone \'https://2965dbeaddd4f53bf03281027f453c69db771392@github.com/neelabh17/MAVI-Face.git\' temp")
        print("MAVI-Face.git Cloned")

        print("The temp folder has been set up")
        print("Rippling changes to temp folder")
        print("#################################################")
        
        os.system("rsync -aP --exclude=\'/weights/*.pth\' --exclude=\'.git\' --exclude=\'widerface_evaluate/**images/**.jpg\' --exclude=\'widerface_evaluate/**imgResults/**.jpg\' --exclude=\'widerface_evaluate/**.txt\' --exclude=\'data/**.jpg\'  \'./Pytorch_Retinaface/\' \'./temp/\'")
        print("Changing directory into temp")
        print("#################################################")
        
        os.chdir("temp")
        print("Entering Github credentials")
        print("#################################################")
        
        os.system("git config --global user.email \'neelabh.madan@outlook.com\'")
        print("#################################################")
        
        os.system("git config --global user.name \'neelabh17\'")
        print("Credentials Entered")
        print("Staging(Add) files")
        print("#################################################")
        
        os.system("git add .")
        print("Commiting files")
        commit_name=input("Enter commit Name")
        print("#################################################")
        
        os.system("git commit -m\'"+commit_name+"\'")
        print("Pushing it onto Github")
        print("#################################################")
        
        os.system("git push origin master")
        print("\nIt has been pushed")


        
    else:
        print("The temp folder already exists")

        print("The temp folder already exists\nDeleting it")
        print("#################################################")
        
        os.system("rm -rf temp")

        print("Temp Doesn't exist creating one and mounting it with the MAVI-Face.git")
        print("#################################################")
        
        os.system("mkdir temp")
        print("Directory Made")
        print("#################################################")
        
        os.system("! git clone \'https://2965dbeaddd4f53bf03281027f453c69db771392@github.com/neelabh17/MAVI-Face.git\' temp")
        print("MAVI-Face.git Cloned")

        print("The temp folder has been set up")

        print("Rippling changes to temp folder")
        print("#################################################")
        
        os.system("rsync -aP --exclude=\'/weights/*.pth\' --exclude=\'.git\' --exclude=\'widerface_evaluate/**images/**.jpg\' --exclude=\'widerface_evaluate/**imgResults/**.jpg\' --exclude=\'widerface_evaluate/**.txt\' --exclude=\'data/**.jpg\'  \'./Pytorch_Retinaface/\' \'./temp/\'")
        print("Changing directory into temp")
        print("#################################################")
        
        os.chdir("temp")
        print("Entering Github credentials")
        print("#################################################")
        
        os.system("git config --global user.email \'neelabh.madan@outlook.com\'")
        print("#################################################")
        
        os.system("git config --global user.name \'neelabh17\'")
        print("Credentials Entered")
        print("Staging(Add) files")
        print("#################################################")
        
        os.system("git add .")
        print("Commiting files")
        commit_name=input("Enter commit Name")
        print("#################################################")
        
        os.system("git commit -m\'"+commit_name+"\'")
        print("Pushing it onto Github")
        print("#################################################")
        
        os.system("git push origin master")
        print("\nIt has been pushed")

elif(n==4):#Get Changes from Github
    print("Changing Directory to cd /content/drive/My Drive/RetinaFace")
    print("#################################################")
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("Checking if temp already exists")
    print("#################################################")
    if not os.path.exists("temp"):
        print("Temp doesn't exist already. lets create it")
        print("Lets put it up there")
        print("Temp Doesn't exist creating one and mounting it with the MAVI-Face.git")
        print("#################################################")
        
        os.system("mkdir temp")
        print("Directory Made")
        print("#################################################")
        
        os.system("! git clone \'https://2965dbeaddd4f53bf03281027f453c69db771392@github.com/neelabh17/MAVI-Face.git\' temp")
        print("MAVI-Face.git Cloned")


    else:
        print("The temp folder already exists")
        print("Changing directory into temp")
        print("#################################################")
        
        os.chdir("temp")
        print("Entering Github credentials")
        print("#################################################")
        
        os.system("git config --global user.email \'neelabh.madan@outlook.com\'")
        print("#################################################")
        
        os.system("git config --global user.name \'neelabh17\'")
        
        print("Lets get a pull request")
        print("#################################################")
        
        os.system("git pull origin master")

    print("Changing directory back")
    print("#################################################")
    
    os.chdir("/content/drive/My Drive/RetinaFace")
    print("temp folder is upto date. Lets update the Pytorch Retinaface folder")
    print("#################################################")
    
    os.system("rsync -aP --exclude=\'.git\' \'./temp/\' \'./Pytorch_Retinaface/\'")



else:
    print("You Havent Selected the right option. Bye!!!")
print("Changing directory back to Retinaface pytorch")
print("#################################################")

os.chdir("/content/drive/My Drive/RetinaFace/Pytorch_Retinaface")