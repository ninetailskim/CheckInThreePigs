import os
import sys

path = sys.argv[1]
print(path)

filelist = os.listdir(path)
print(filelist)

'''
for index, file in enumerate(filelist):
    usedname = path + file
    newname = path + str(index) + file.split('.')[-1]
    os.rename(usedname, newname)
'''