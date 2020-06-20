import numpy as np

a = np.array([[1,2],[3,4],[5,6]])
b = np.array([[1],[2],[3]])
c = np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,0],[11,10]]])

print(a.shape)
print(b.shape)
print(c.shape)

# abc = np.stack((a,b,c), axis=0)

# print(abc)

abc = zip(a,b,c)
labc = list(abc)
for aa in labc:
    print(aa)

import random
random.shuffle(labc)
for aa in labc:
    print(aa)