from toolbox.pickleOpers import loadup

def bestConf(prFilepath):
    prdata=loadup(prFilepath)

    maxF1=0
    confForF1=0
    confForRecall=0
    maxr=0
    maxp=0
    confForPrecision=0
    for data in data:
        # data[0]- prec
        # data[1]-recall
        # data[2]-conf core
        p,r,c=data
        if(p>0.8 and r>0.8):
            if(p>maxp):
                confForPrecision=c
                maxp=p
            if(r>maxr):
                confForRecall=c
                maxr=r
        
        f1=2*r*p/(r+p)
        if(f1>maxF1):
            maxF1=f1
            confForF1=c

    print("For max recall conf= {}".format(confForRecall))
    print("For max precision conf= {}".format(confForPrecision))
    print("For max f1 conf= {}".format(confForF1))
