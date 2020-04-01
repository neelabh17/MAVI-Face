print("Enter new token")
token=input()
print("Changing token")
import os
myCommand="git remote set-url origin https://{}@github.com/neelabh17/MAVI-Face.git".format(token)

os.system(myCommand)