import os 
def makeFake(seriesName,begin,end,step):
    i=begin
    while(i<=end):
        os.system('touch {}_epoch_{}.pth'.format(seriesName,i))

        i+=step