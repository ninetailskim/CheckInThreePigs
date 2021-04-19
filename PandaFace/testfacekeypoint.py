import paddlehub as hub
import cv2
import numpy as np
import math
import os
import glob
import copy


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        self.LU = LandmarkUtils()

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

def constrast_img(img1, con=2.2, bri=3):
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

def main():
    PF = PandaFace()
    FC = FaceCut()
    filelist = glob.glob("")

    s = 0
    cons = []
    while s < 5:
        cons.append(s)
        s += 0.2

    os.makedirs("res", exist_ok=True)

    for file in filelist:
        basename = os.path.basename(file)
        img = cv2.imread(file)
        mask, frame, top, bottom, left, right = FC.getFaceByLandmark(img)

        kernel = np.ones((3,3), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=10)

        back = np.ones_like(frame) * 255
        res = (mask / 255) * frame + (1-(mask / 255)) * back
        res = res.astype(np.uint8)
        tres = copy.deepcopy(res)
        for con in cons:
            image = constrast_img(res, con, 3)
            tres = np.concatenate([tres, image], axis=1)
        for bri in [0,1,2]:
            image = constrast_img(res, 2.2, bri)
            tres = np.concatenate([tres, image], axis=1)
        cv2.imwrite("res/"+basename, tres)
        

if __name__ == '__main__':
    main()