import cv2
import random

img = cv2.imread("images/0_0.png")
cv2.imshow("test",img)
cv2.waitKey()
print(img.shape)
img = img[1:,:]
print(img.shape)
img = cv2.copyMakeBorder(img, 0,1,0,0, cv2.BORDER_REPLICATE)
print(img.shape)

img = img[:,1:]
print(img.shape)
img = cv2.copyMakeBorder(img, 0,0,0,1, cv2.BORDER_REPLICATE)
print(img.shape)

img = img[:-1,:]
print(img.shape)
img = cv2.copyMakeBorder(img, 1,0,0,0, cv2.BORDER_REPLICATE)
print(img.shape)

img = img[:,:-1]
print(img.shape)
img = cv2.copyMakeBorder(img, 0,0,1,0, cv2.BORDER_REPLICATE)
print(img.shape)


img[10:20,15:28] = 255
cv2.imshow("test",img)
cv2.waitKey()

for i in range(10):
    print(random.randint(0,1))

