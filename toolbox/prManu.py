from toolbox.pickleOpers import loadup

def bestConf(prFilepath,atleastP=0.8,atleastR=0.8):
    prdata=loadup(prFilepath)

    maxF1=0
    confForF1=0
    confForRecall=0
    maxr=0
    maxp=0
    confForPrecision=0
    for data in prdata:
        # data[0]- prec
        # data[1]-recall
        # data[2]-conf core
        p,r,c=data
        if(p>atleastP and r>atleastR):
            if(p>maxp):
                confForPrecision=data
                maxp=p
            if(r>maxr):
                confForRecall=data
                maxr=r
        
        f1=2*r*p/(r+p)
        if(f1>maxF1):
            maxF1=f1
            confForF1=data

    print("For max recall conf\n[precision,recall,conf]\n {}\n-------------".format(confForRecall))
    print("For max precision conf\n[precision,recall,conf]\n {}\n-------------".format(confForPrecision))
    print("For max f1 conf\n[precision,recall,conf]\n {}\n-------------".format(confForF1))
    a="[precision,recall,conf]\n {}\n-------------".format(confForRecall)
    b="[precision,recall,conf]\n {}\n-------------".format(confForPrecision)
    c="[precision,recall,conf]\n {}\n-------------".format(confForF1)

    return a,b,c
