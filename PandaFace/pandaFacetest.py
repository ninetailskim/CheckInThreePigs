import paddlehub as hub
import cv2
import numpy as np
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class segUtils():
    def __init__(self):
        super().__init__()
        self.module = hub.Module(name="ace2p")

    def predict(self, frame):
        result = self.module.segmentation(images=[frame], use_gpu=True)
        result = result[0]['data']
        result[result != 13] = 0
        result[result == 13] = 1
        return result

class detUtils():
    def __init__(self):
        super(detUtils, self).__init__()
        self.module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
        self.last = None

    def distance(self, a, b):
        return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

    def iou(self, bbox1, bbox2):

        b1left = bbox1['left']
        b1right = bbox1['right']
        b1top = bbox1['top']
        b1bottom = bbox1['bottom']

        b2left = bbox2['left']
        b2right = bbox2['right']
        b2top = bbox2['top']
        b2bottom = bbox2['bottom']

        area1 = (b1bottom - b1top) * (b1right - b1left)
        area2 = (b2bottom - b2top) * (b2right - b2left)

        w = min(b1right, b2right) - max(b1left, b2left)
        h = min(b1bottom, b2bottom) - max(b1top, b2top)

        dis = self.distance([(b1left+b1right)/2, (b1bottom+b1top)/2],[(b2left+b2right)/2, (b2bottom+b2top)/2])

        if w <= 0 or h <= 0:
            return 0, dis
        
        iou = w * h / (area1 + area2 - w * h)
        return iou, dis

    def predict(self, frame):
        res = self.module.face_detection(images=[frame], use_gpu=True)
        reslist = res[0]['data']
        if len(reslist) == 0:
            if self.last is not None:
                return self.last
            else:
                return None
        elif len(reslist) == 1:
            self.last = reslist[0]
            return reslist[0]
        else:
            maxiou = -float('inf')
            maxi = 0
            mind = float('inf')
            mini = 0
            for index in range(len(reslist)):
                tiou, td = self.iou(self.last, reslist[index])
                if tiou > maxiou:
                    maxi = index
                    maxiou = tiou
                if td < mind:
                    mind = td
                    mini = index  
            if tiou == 0:
                self.last = reslist[mini]
                return reslist[mini]
            else:
                self.last = reslist[maxi]
                return reslist[maxi]

class LandmarkUtils():
    def __init__(self):
        super().__init__()
        self.module = hub.Module(name="face_landmark_localization")

    def predict(self, frame):
        result = self.module.keypoint_detection(images=[frame], use_gpu=True)
        if result is not None:
            return result[0]['data']
        else:
            return None


class FaceCut():
    def __init__(self):
        super().__init__()
        self.SU = segUtils()
        self.DU = detUtils()
        self.LU = LandmarkUtils()

    def getFace(self, frame):
        dres = self.DU.predict(frame)
        sres = self.SU.predict(frame)

        if dres is None:
            return None
        else:
            top = int(dres['top'])
            left = int(dres['left'])
            right = int(dres['right'])
            bottom = int(dres['bottom'])
        
            sres3 = np.repeat(sres[:,:,np.newaxis], 3, axis=2)
            facecut = sres3 * frame
            return facecut[top:bottom,left:right], sres3[top:bottom,left:right]

    def getFaceByLandmark(self, frame):
        result = self.LU.predict(frame)
        if result is None:
            return None, None
        else:
            #print(result)
            #print(len(result))
            result = result[0]
            #print(result)
            #print(len(result))
            mask = np.zeros_like(frame).astype(np.uint8)
            pts = []
            order = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,27,26,25,20,19,18]
            
            h,w = frame.shape[:2]
            top = h
            bottom = 0
            left = w
            right = 0

            for o in order:
                tx = int(result[o-1][0])
                ty = int(result[o-1][1])
                if tx < left:
                    left = tx
                if tx > right:
                    right = tx
                if ty < top:
                    top = ty
                if ty > bottom:
                    bottom = ty
                pts.append([tx, ty])
            mask = cv2.fillPoly(mask, [np.array(pts)], (255, 255, 255)).astype(np.uint8)

            pframe = cv2.polylines(frame, [np.array(pts)], True, (255, 255, 255))

            return mask[top:bottom, left:right], pframe[top:bottom, left:right]


def constrast_img(img1, c, b):
    rows, cols, channels = img1.shape

    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    grey = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    O = cv2.equalizeHist(grey)
    img = np.concatenate([img1, dst], axis=1)
    img2 = np.concatenate([grey, O], axis=1)
    return img , img2   


FC = FaceCut()

img = cv2.imread("1.jpg")
h, w = img.shape[:2]

img = cv2.resize(img, (int(w/4), int(h/4)))


face, facemask = FC.getFace(img)

back = np.ones_like(face) * 255
oface = face * facemask + back * (1 - facemask)

oface = oface.astype(np.uint8)

# cv2.imshow("face", oface)
# cv2.waitKey(0)

facemask = cv2.GaussianBlur(facemask.astype(np.float32), (25,25), 0)
facemask = np.floor(facemask)

gface = face * facemask + back * (1 - facemask)

gface = gface.astype(np.uint8)

# cv2.imshow("Gaussianface", gface)
# cv2.waitKey(0)

grayface = cv2.cvtColor(gface, cv2.COLOR_BGR2GRAY)

# cv2.imshow("grayface", grayface)
# cv2.waitKey(0)

# print(grayface.shape)

# for i in range(50,200,10):
#     img_binary = cv2.threshold(grayface, i, 255, cv2.THRESH_BINARY)[1]
#     cv2.imshow("neggrayface", img_binary)
#     cv2.waitKey(0)

mask, pframe = FC.getFaceByLandmark(img)
cv2.imshow("mask", mask)
cv2.waitKey(0)

cv2.imshow("pframe", pframe)
cv2.waitKey(0)

back = np.ones_like(pframe) * 255

res = (mask/255) * pframe + (1-(mask/255)) * back
res = res.astype(np.uint8)
cv2.imshow("res", res)
cv2.waitKey(0)

s = 0
cons = []
while s < 5:
    cons.append(s)
    s += 0.2

for con in cons:
    print("constract: ", con," brightness: ", 3)
    image, image2 = constrast_img(res, con, 3)
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.imshow("testg", image2)

cv2.destroyAllWindows()