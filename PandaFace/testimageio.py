import imageio
import cv2


img = imageio.imread("1.jpg")
cvimg = cv2.imread("1.jpg")

cv_img = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)

cv2.imshow("imageio", cv_img)
cv2.waitKey(0)


cv2.imshow("cv2", cvimg)
cv2.waitKey(0)

print(type(img))