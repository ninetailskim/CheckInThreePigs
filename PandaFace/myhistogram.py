import cv2
import numpy as np

def histMatch(hista, tpl):
    j = 1
    res = np.zeros_like(hista)
    for i in range(256):
        #print(i,hista[i][0],"---",j,tpl[j][0])
        while j < 255 and hista[i][0] > tpl[j][0]:
            j += 1 
        if abs(hista[i][0] - tpl[j][0]) < abs(hista[i][0] - tpl[j - 1][0]):
            res[i][0] = j
        else:
            res[i][0] = j - 1
    res = np.reshape(res, [256]).astype(np.uint8)
    return res

class myHistogram():
    def __init__(self, img):
        super().__init__()
        self.h, self.w = img.shape
        self.hist = self.calhistogram(img) / (self.h * self.w)
        self.p = self.hist2p()

    def calhistogram(self, image):
        return cv2.calcHist([image],[0],None, [256],[0,256])

    def hist2p(self):
        p = np.zeros_like(self.hist)
        sum = 0
        for i in range(256):
            t = self.hist[i][0]
            sum += t
            p[i][0] = sum
        return p




img = cv2.imread("1.jpg")
h, w = img.shape[:2]
img = cv2.resize(img, (int(w/4), int(h/4)))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mh = myHistogram(gray)

img2 = cv2.imread("pf1.jpg")
h, w = img2.shape[:2]
img2 = cv2.resize(img2, (int(w/4), int(h/4)))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

mh2 = myHistogram(gray2)

lut = histMatch(mh.p, mh2.p)

print(lut)
lut = np.reshape(lut, [256]).astype(np.uint8)
print(lut)
print(lut.shape)

cv2.imshow("origin", gray)
cv2.waitKey(0)

res = cv2.LUT(gray, lut)

cv2.imshow("template", gray2)
cv2.waitKey(0)

cv2.imshow("result", res)
cv2.waitKey(0)

cv2.destroyAllWindows()
