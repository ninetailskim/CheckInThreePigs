import cv2
import numpy as np

def constrast_img(img1, c, b):
    rows, cols, channels = img1.shape

    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    grey = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    O = cv2.equalizeHist(grey)
    img = np.concatenate([img1, dst], axis=1)
    img2 = np.concatenate([grey, O], axis=1)
    return img , img2

img = cv2.imread("1.jpg")

h, w = img.shape[:2]

img = cv2.resize(img, (int(w/4), int(h/4)))

pic = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("g", grey)
cv2.waitKey(0)

O = cv2.equalizeHist(grey)
cv2.imshow("o", O)
cv2.waitKey(0)

s = 0
cons = []
while s < 5:
    cons.append(s)
    s += 0.2

for con in cons:
    image, image2 = constrast_img(img, con, 3)
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.imshow("testg", image2)

cv2.destroyAllWindows()