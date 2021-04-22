import numpy as np
import cv2


a = np.array([1,2,3,3,4,5,6,7,8,9,10])

print(a)

a[a < 5] -= 3
a[a >= 5] += 5

print(a)

a = np.clip(a, 0, 10)
print(a)

img = cv2.imread("1.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


