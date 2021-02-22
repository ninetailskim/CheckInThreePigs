import paddlehub as hub
import cv2
import numpy as np
import pygame as pg
import time
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

currentSeg = None
currentTime = 0
genTime = []
currentIndex = 0
gm = []
W = 0
H = 0
showimg = None
minPIXEL = 30 * 30
dangerousPIXEL = 50 * 50

class segUtils:
    def __init__(self):
        super(segUtils, self).__init__()
        self.ace2p = hub.Module(name='ace2p')

    def getMask(self, frame):
        res = ace2p.segmentation([frame], use_gpu=True)
        if isinstance(res, list):
            resint = res[0]['data']
            resint[resint != 2] = 0
            resint[resint == 2] = 1
            return resint
        else:
            return None

def getPIXEL(x, y, radius):
    t = y - radius if y - radius > 0 else 0
    l = x - radius if x - radius > 0 else 0
    b = y + radius if y + radius < H else H
    r = x + radius if x + radius < W else W
    return t,l,b,r

class Ball:
    x = None
    y = None
    speed_x = None
    speed_y = None
    radius = None
    color = None

    def __init__(self, x, y, speed_x, speed_y, radius, color):
        self.x = x
        self.y = y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.radius = radius
        self.color = color

    def draw(self, screen):
        t,l,b,r = getPIXEL(self.x, self.y, self.radius)
        if isinstance(self.color, int):
            showimg[t:b,l:r,:] = self.color
        else:
            showimg[t:b,l:r,0] = self.color[0]
            showimg[t:b,l:r,1] = self.color[1]
            showimg[t:b,l:r,2] = self.color[2]
        # pg.draw.circle(screen, self.color, [self.x, self.y], self.radius)
        

    def move(self, screen):
        self.x += self.speed_x
        self.y += self.speed_y
        
        if self.x > W - self.radius or self.x < self.radius:
            self.speed_x = -self.speed_x

        if self.y > H - self.radius or self.y < self.radius:
            self.speed_y = -self.speed_y

        time.sleep(0.001)

        self.draw(screen)

balls = []

def randomXY():
    x = random.randint(0, W)
    y = random.randint(0, H)
    return x, y

def inseg(x, y, radius):
    if currentSeg is None:
        return False
    else:
        t,l,b,r = getPIXEL(x, y, radius)
        if np.sum(currentSeg[t:b,b:r]) > 0:
            return True
        else:
            return False


def create_ball(screen):

    r = 3
    color = 0

    x, y = randomXY()
    if inseg(x,y,r):
        x, y = randomXY()

    speed_x = random.randint(-5, 5)
    speed_y = random.randint(-5, 5)
    
    b = Ball(x, y, speed_x, speed_y, r, color)
    balls.append(b)
    # b.draw(screen) 


def ball_manager():
    if currentIndex <= len(gm):
        if currentTime < genTime[currentIndex]:
            for i in range(gm[currentIndex]):
                create_ball()
        else:
            currentIndex += 1
    
    for b in balls:
        b.move(showimg)

def main():
    cap = cv2.VideoCapture(0)
    su = segUtils()
    while True:
        ret, frame = cap.read()
        if ret == True:
            H, W = frame.shape[:2]
            
            currentSeg = su.getMask(frame)
            if currentSeg is not None:
                if np.sum(currentSeg) < minPIXEL:
                    if showimg is None:
                        showimg = np.ones_like(frame) * 255
                    #打字
                    cv2.imshow('Game', showimg)
                    cv2.waitKey(0)

                else:
                    showimg = np.ones_like(frame) * 255
                    currentSeg3 = np.repeat(currentSeg[:,:,np.newaxis], 3, axis=2)
                    if np.sum(currentSeg) < dangerousPIXEL:
                        frame[:,:,2] = 255
                    showimg = frame * currentSeg3 + showimg * (1 - currentSeg3)
                    ball_manager()
                    showimg = showimg.astype(np.uint8)
                    cv2.putText()
                    cv2.imshow('Game', showimg)
                    cv2.waitKey(1)
        else:
            print("Check your camera!")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()