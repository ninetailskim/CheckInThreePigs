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
        #dres = self.DU.predict(frame)
        sres = self.SU.predict(frame)
        
        sres3 = np.repeat(sres[:,:,np.newaxis], 3, axis=2)
        return sres3

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

            # pframe = cv2.polylines(frame, [np.array(pts)], True, (255, 255, 255))

            return mask[top:bottom, left:right], frame[top:bottom, left:right], top, bottom, left, right

class PandaFace():
    def __init__(self):
        super().__init__()
        self.FC = FaceCut()

    def memecut(self, meme):
        mask = self.FC.getFace(meme)

        cv2.imshow("mask", mask * 255)
        cv2.waitKey(0)

        kernel = np.ones((3,3), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        ones255 = np.ones_like(mask) * 255
        meme = (ones255 * mask + meme * (1 - mask)).astype(np.uint8)
        return meme

    def constrast_img(self, img1, con=2.2, bri=3):
        rows, cols, channels = img1.shape
        blank = np.zeros([rows, cols, channels], img1.dtype)
        dst = cv2.addWeighted(img1, con, blank, 1-con, bri)
        grey = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        #O = cv2.equalizeHist(grey)
        #img = np.concatenate([img1, dst], axis=1)
        #img2 = np.concatenate([grey, O], axis=1)
        #return img , img2 
        #grey = cv2.equalizeHist(grey)
        grey =  cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        return grey 

    def facecut(self, face):
        mask, face, _, _, _, _ = self.FC.getFaceByLandmark(face)
        back = np.ones_like(face) * 255
        res = (mask / 255) * face + (1-(mask / 255)) * back
        res = res.astype(np.uint8)
        image = self.constrast_img(res)
        return image, mask

    def compose(self, meme, face):
        _, _, top, bottom, left, right = self.FC.getFaceByLandmark(meme)
        meme = self.memecut(meme)
        face,mask = self.facecut(face)
        h,w = face.shape[:2]
        mh = bottom - top - 10
        mw = right - left - 10
        # print(mh, mw)
        cx = int((right + left) / 2)
        cy = int((top + bottom) / 2)
        neww = mw
        newh = mh
        if h/w < mh/mw:
            #w
            neww = mw
            newh = int(neww / w * h)
            face = cv2.resize(face, (neww, newh))
            mask = cv2.resize(mask, (neww, newh))
        else:
            #h
            newh = mh
            neww = int(newh / h * w)
            face = cv2.resize(face, (neww, newh))
            mask = cv2.resize(mask, (neww, newh))
        # print(newh, neww)
        cx -= int(neww / 2)
        cy -= int(newh / 2)
        cv2.imshow("meme", meme.astype(np.uint8))
        cv2.waitKey(0)
        meme[cy:cy+newh, cx:cx+neww] = face * (mask / 255) + meme[cy:cy+newh, cx:cx+neww] * (1 - mask / 255)
        meme.astype(np.uint8)
        return meme

def main():
    PF = PandaFace()
    meme = cv2.imread("origin\\p4.jpeg")
    face = cv2.imread("1.jpg")

    meme = PF.compose(meme, face)
    cv2.imshow("res", meme)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()