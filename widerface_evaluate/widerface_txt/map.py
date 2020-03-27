a=open("Resnet50_epoch_{}_noGrad_FT/results.txt".format(2))

lines=a.readlines()
for line in lines:
    print(line)