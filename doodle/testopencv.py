import numpy as np
import cv2


img = np.ones([500,500,3]) * 255
cv2.line(img, pt1=(-10,-10), pt2=(250,250), color=(0,0,0), thickness=5)
cv2.line(img, pt1=(-10,250), pt2=(250,250), color=(255,0,0), thickness=5)
cv2.line(img, pt1=(250,-10), pt2=(250,250), color=(0,0,255), thickness=5)
cv2.line(img, pt1=(250,-10), pt2=(750,250), color=(0,255,0), thickness=5)
cv2.circle(img,(200, 350), 8, (0,0,255), -1)
cv2.imshow("ss", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("e.jpg")
h,w = img.shape[:2]
img = cv2.resize(img, (int(w/3), int(h/3)))
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("template2.jpg",img)
