import os
import sys

path = sys.argv[1]
pre = sys.argv[2]
print(path)

filelist = os.listdir(path)
print(filelist)

print(len(filelist))


for index, file in enumerate(filelist):
    usedname = os.path.join(path, file)
    #newname = os.path.join(path, pre+"_"+str(index)+file.split('.')[-1])
    newname = os.path.join(path, pre+"_"+str(index)+'.'+file[-3:])
    os.rename(usedname, newname)
