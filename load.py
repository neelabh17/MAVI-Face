#landms,loc, conf = face_detector.infer(frame)
#loc:  (1, 16800, 2)
#conf:  (1, 16800, 4)
#landms:  (1, 16800, 10)

import pickle
model_name="faceDetector"
def load():
    # for reading also binary mode is important 
    dbfile = open(model_name, 'rb')      
    db = pickle.load(dbfile) 
    for keys in db: 
        print(keys, '=>', db[keys].shape) 
    # k=0
    # for a in db["loc"][0]:
    #     if(a[1]>0.5):
    #         # k=a[1]
    #         print(a)
    # print(a)
    # print("----------------"+str(k))
    dbfile.close()

    print(db["conf"])
 
load()



