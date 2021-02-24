import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import cv2
import paddlehub as hub
import numpy as np
import math
from PIL import Image
from sklearn.cluster import KMeans

module = hub.Module(name="face_landmark_localization")
debug = True
use_gpu = False

class KMEANSEG(object):
    """
    Seg image by kmeans
    """
    def __init__(self, initframe):
        super(self, KMEANSEG).__init__()
        self.kmeans = KMeans(n_clusters=2,random_state=0)
        self.initKMeans(initframe)

    def initKMeans(self, frame):
        recomimg = self.preprocess(frame)
        self.kmeans.fit(recomimg)

    def __call__(self, frame):
        h, w, recomimg = self.preprocess(frame)
        labels = self.kmeans.pridict(recoming)
        return self.postprocess(labels, h, w)

    def preprocess(self, frame):
        h,w,_ = frame.shape

        recomimg = np.concatenate([frame.astype(np.int32),
                                   np.repeat(np.arange(w)[np.newaxis,:], h, axis=0)[:,:,np.newaxis],
                                   np.repeat(np.arange(h)[:,np.newaxis], w, axis=1)[:,:,np.newaxis]],
                                  axis=2).reshape([-1, 5])

        return h, w, recomimg

    def postprocess(self, labels, h, w):
        mask = np.repeat(labels.reshape([h,w,1]), 3, axis=2)
        blurmask = cv2.blur(mask.astype(np.float32), (5,5))
        blurmask = np.ceil(blurmask).astype(np.uint8)
        return blurmask

class Compose(object):
    """
    compose background with hair by PIL paste
    """
    def __init__(self, frame, resource='hair.mp4'):
        self.KMS = KMEANSEG(frame)
        self.hair = []
        self.preprocesshair()
        self.haircap = cv2.VideoCapture(resource)
        self.index = 0

    def preprocesshair(self):
        self.framecount = int(self.haircap.get(cv2.CAP_PROP_FRAME_COUNT))
        while self.haircap.isopened():
            ret, frame = self.haircap.read()
            if not ret:
                break
            else:
                self.hair.append(frame)

    def compose(self, back, pointl, pointr):
        tanangle = 1.0 * (pointl[0] - pointr[0]) / (pointl[1] - pointr[1])
        angle = math.atan(tanangle)  / math.pi * 180
        distance = math.sqrt((pointl[0] - pointr[0]) ** 2 + (pointl[1] - pointr[1]) ** 2)
        self.index = self.index % len(self.hair)
        
        PILhair = Image.fromarray(cv2.cvtColor(self.hair[self.index], cv2.COLOR_BGR2RGB))
        ROTATEhair = PILhair.rotate(angle, expand=True)
        rotatehair = cv2.cvtColor(np.asarray(ROTATEhair),cv2.COLOR_RGB2BGR)
        rotatemask = self.KMS(rotatehair)
        





        self.index += 1

def get_point(module, frame):
    result = module.keypoint_detection(images=[frame], use_gpu=use_gpu)
    if isinstance(result, list):
        return True, result[0]['data'][0]
    else:
        return False, None

def getHair():
    haircap = cv2.VideoCapture('hair.mp4')
    framecount = int(haircap.get(cv2.CAP_PROP_FRAME_COUNT))
    while haircap.isopened():
        ret, frame = haircap.read()
        if not ret:
            break
        frame



def getHairAndMask(angle):


resultname = "camera.mp4"
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(resultname, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    success, face_landmark = get_point(module, frame)

    if not success:
        pass
    else:
        if debug:
            tmp_img = frame.copy()
            for _, point in enumerate(face_landmark):
                cv2.circle(tmp_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)      
            
            cv2.imshow("Debug", tmp_img)
            cv2.waitKey(1)
        pointl = face_landmark[0]
        pointr = face_landmark[16]
        tanangle = 1.0 * (pointl[0] - pointr[0]) / (pointl[1] - pointr[1])
        angle = math.atan(tanangle)  / math.pi * 180
        distance = math.sqrt((pointl[0] - pointr[0]) ** 2 + (pointl[1] - pointr[1]) ** 2)
        getHairAndMask(angle)

