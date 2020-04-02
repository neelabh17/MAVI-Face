
the pickle file contains the dictionary


how to load

a=open("label.pickle","rb")
import pickle
data=pickle.load(a)


now 

for a in data:
	print(a) --------------->>>>>>>>>>> gives out the file path NJIS/imagexyz.jpg
	print(data[a]) ------------------>>>>>>>>>>> gives a nX(4+ something dimentional for us only the first 4 are important)


so you basicaly get it right no