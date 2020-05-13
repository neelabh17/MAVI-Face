import os

def make(dirName):
    # dirname should be a directory name not a file name
    if(not os.path.isdir(dirName)):
        os.makedirs(dirName)