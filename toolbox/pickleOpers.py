import pickle
def save(data,fileLocation):
    f=open(fileLocation,"wb")
    pickle.dump(data,f)
    f.close()

def loadup(fileLocation):
    dataFile=open(fileLocation,"rb")
    data=pickle.load(dataFile)
    dataFile.close()
    return data