import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9,0])
b = np.array([1,2,3,4,5,6,7,8,9,0])
c = np.array([1,2,3,4,5,6,7,8,9,0])

d = []

d.extend(a)
d.extend(b)

#x = np.vstack(a,b,c)

#print(x)
print(d)

x = np.stack((a,b,c),axis=0)

print(x)