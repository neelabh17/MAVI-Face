import os

def make(folderName):
    if(not os.path.isdir(folderName)):
        os.makedirs(folderName)