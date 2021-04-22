import paddlehub as hub
import cv2
import numpy as np
import math
import os
import glob
import copy
import sys


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
        self.SU = segUtils()

    def getFace(self, frame):
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

            mask2 = self.getFace(frame)

            mask = mask * mask2 

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
    grey = grey.astype(np.int32)
    grey[grey < 80] -= 50
    grey[grey >= 80] += 50
    grey = np.clip(grey, 0, 255)
    grey = grey.astype(np.uint8)
    grey =  cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    return grey 

def gamma_img(img1, gamma):
    grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    out = cv2.LUT(grey, gamma_table)
    # print(img1.shape)
    # print(grey.shape)
    # print(out.shape)
    out = out.astype(np.int32)
    out[out < 80] -= 50
    out[out >= 80] += 50
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    grey =  cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return grey

def main():
    PF = PandaFace()
    FC = FaceCut()
    filelist = glob.glob("*.jpg")
    

    s = 0
    cons = []
    while s < 5:
        cons.append(s)
        s += 0.2

    os.makedirs(sys.argv[1], exist_ok=True)

    for file in filelist:
        print(file)
        basename = os.path.basename(file)
        img = cv2.imread(file)

        h,w, _ = img.shape
        while max(h, w) > 2000:
            h /= 2
            w /= 2
        
        img = cv2.resize(img, (int(w), int(h)))

        mask, frame, _, _, _, _ = FC.getFaceByLandmark(img)

        kernel = np.ones((3,3), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=10)

        back = np.ones_like(frame) * 255
        res = (mask / 255) * frame + (1-(mask / 255)) * back
        res = res.astype(np.uint8)
        tres = copy.deepcopy(res)
        for con in cons:
            image = constrast_img(res, con, 3)
            tres = np.concatenate([tres, image], axis=1)
        cv2.imwrite(sys.argv[1]+"/cons_"+basename, tres)
        tres = copy.deepcopy(res)
        for bri in [0,20,40]:
            image = constrast_img(res, 2.2, bri)
            tres = np.concatenate([tres, image], axis=1)
        for bri in [0,50,100]:
            image = constrast_img(res, 2.2, bri)
            tres = np.concatenate([tres, image], axis=1)
        cv2.imwrite(sys.argv[1]+"/brig_"+basename, tres)
        tres = copy.deepcopy(res)
        for gamma in [0.1, 0.2, 0.4, 0.67, 1.5, 2.5, 5, 10, 25]:
            image = gamma_img(res, gamma)
            tres = np.concatenate([tres, image], axis=1)
        cv2.imwrite(sys.argv[1]+"/gamm_"+basename, tres)
      

if __name__ == '__main__':
    main()