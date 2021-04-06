import cv2
import numpy as np

x = np.zeros((200,200,3), dtype=np.uint8)
y = np.zeros((200,200,3), dtype=np.uint8)
pts = np.array([[0.0,100.01],[100,200],[200,100],[100,0]])

cv2.fillPoly(x, [pts], (255,255,255))

cv2.imshow("xtest", x)
cv2.waitKey(0)

cv2.polylines(y, [pts], True, (255,255,255))
cv2.imshow("xtest", y)
cv2.waitKey(0)
cv2.destroyAllWindows()