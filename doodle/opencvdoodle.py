import cv2
import numpy as np
import doodle
import paddlehub as hub
import os
import copy
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"


isMouseLBDown = False
circleColor = (0, 0, 0)
circleRadius = 5
lastPoint = (0, 0)

lines = []
colors = []
cirRads = []

drawNewLine = True

class estUtil():
    def __init__(self):
        super(estUtil, self).__init__()
        self.module = hub.Module(name='human_pose_estimation_resnet50_mpii')

    def do_est(self, frame):
        res = self.module.keypoint_detection(images=[frame], use_gpu=True)
        return res[0]['data']

def drawOnCanvas(canvas, skins):
    for skin in skins:
        if skin.init:
            pos = skin.getPos()
            lp = (int(pos[0]), int(pos[1]))
        else: 
            pos = skin.getPos()
            cv2.line(canvas, pt1=lp, pt2=(int(pos[0]), int(pos[1])), color=skin.color, thickness=skin.cirRad)
            lp = (int(pos[0]), int(pos[1]))
            # cv2.imshow("test",canvas)
            # cv2.waitKey(0)

def draw_circle(event, x, y, flags, param):

    global img
    global isMouseLBDown
    global lastPoint
    if drawNewLine is True:
        if event == cv2.EVENT_LBUTTONDOWN:
            isMouseLBDown = True
            cv2.circle(img, (x, y), int(circleRadius/2), circleColor, -1)
            lastPoint = (x, y)
            lines.append([(x,y)])
            colors.append(circleColor)
            cirRads.append(circleRadius)
        elif event == cv2.EVENT_LBUTTONUP:
            isMouseLBDown = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if isMouseLBDown:
                if lastPoint is not None:
                    cv2.line(img, pt1=lastPoint, pt2=(x,y), color=circleColor, thickness=circleRadius)
                    lastPoint = (x, y)
                    lines[-1].append((x,y))


def updateCircleColor(x):
    global circleColor
    # global colorPreviewImg
    r = cv2.getTrackbarPos('Channel_Red', 'Doodle')
    g = cv2.getTrackbarPos('Channel_Green', 'Doodle')
    b = cv2.getTrackbarPos('Channel_Blue', 'Doodle')
    circleColor = (b, g, r)
    # colorPreviewImg[:] = circleColor

def updateCircleRadius(x):
    global circleRadius
    # global radiusPreview
    circleRadius = cv2.getTrackbarPos('Circle_Radius', 'Doodle')
    # radiusPreview[:] = (255, 255, 255)
    # cv2.circle(radiusPreview, center=(50, 50), radius=int(circleRadius / 2), color=(0,0,0), thickness=-1)

def drawTemplate(canvas, eu):
    img = cv2.imread("template2.jpg")
    th, tw = img.shape[:2]
    oh, ow = canvas.shape[:2]
    ih = (oh - th) / 2
    iw = (ow - tw) / 2
    tres = eu.do_est(img)
    for key, value in tres.items():
        tres[key] = [int(value[0] + iw), int(value[1] + ih)]
        cv2.circle(canvas,(tres[key][0], tres[key][1]), 5, (0,0,255), -1)

    return tres

cv2.namedWindow('Doodle')

cv2.createTrackbar('Channel_Red', 'Doodle', 0, 255, updateCircleColor)
cv2.createTrackbar('Channel_Green', 'Doodle', 0, 255, updateCircleColor)
cv2.createTrackbar('Channel_Blue', 'Doodle', 0, 255, updateCircleColor)
cv2.createTrackbar('Circle_Radius', 'Doodle',1, 20, updateCircleRadius)

cv2.setMouseCallback('Doodle', draw_circle)

drawMode = True

videoStream = "mabaoguo.mp4"
# videoStream = 0



cap = cv2.VideoCapture(videoStream)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
img = np.ones((height, width, 3)) * 255

eu = estUtil()

templatekeypoint = drawTemplate(img, eu)

initmatch = None
skins = []

drawOrigin = True

while(True):
    if drawNewLine:
        cv2.imshow('Doodle', img)
        
    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('c'):
        drawNewLine = not drawNewLine
        isMouseLBDown = not isMouseLBDown
        lastPoint= None
        initmatch = True
        #丰富关键点,增加中点addcenterPoint
        ckeypoint = doodle.addcenterPoint(templatekeypoint)
        ack, fas = doodle.complexres(ckeypoint, copy.deepcopy(doodle.FatherAndSon))
        ack, fas = doodle.complexres(ack, fas)
        print("complexres")
        #把关键点转成nodeItem格式,也就是toNodes
        nodes = doodle.toNodes(ack, fas)
        print("toNodes")
        print("nodes: ", len(nodes))
        #建立node之间的父子关系
        nodes = doodle.connectNodes(nodes, fas)
        print("connectNodes")
        print("nodes: ", len(nodes))
        #计算每个结点的角度信息setinfo
        doodle.setInfo(nodes['centerpoint'])
        print("setInfo")
        print("nodes: ", len(nodes))
        #buildskin 在每次切换到drawline之后只计算一次
        # doodle.debugNodes(nodes, np.ones((height * 3, width * 3, 3)) * 255)
        skins = doodle.buildskin(lines, colors, cirRads, nodes)

        # doodle.debug(skins,np.ones((height * 3, width * 3, 3)) * 255)
        print("buildskin")
        time.sleep(10)
        
    if drawNewLine is False:
        ret, frame = cap.read()
        if ret == True:
            if videoStream == 0:
                frame = cv2.flip(frame, 1)
            # cv2.imshow("test", frame)
            # cv2.waitKey(0)

            keypoint = eu.do_est(frame)
            #计算scale
            scale = doodle.distance(keypoint['thorax'],keypoint['pelvis']) / doodle.distance(templatekeypoint['thorax'],templatekeypoint['pelvis'])
            #丰富关键点,增加中点addcenterPoint
            ckeypoint = doodle.addcenterPoint(keypoint)
            ack, fas = doodle.complexres(ckeypoint, copy.deepcopy(doodle.FatherAndSon))
            ack, _ = doodle.complexres(ack, fas)
            #update nodes的POS和info#其实主要是node的x,y,thabs
            # doodle.debugNodesInfo(nodes, copy.deepcopy(doodle.FatherAndSon), skins)
            doodle.updateNodesXY(nodes, ack)
            doodle.setInfo(nodes['centerpoint'])
            #calculateSkin计算新的皮肤的位置
            tskins = doodle.calculateSkin(copy.deepcopy(skins), 1)
            # newimg每帧新生成,然后都画到newimg上去
            if drawOrigin == True:
                moveimg = frame 
            else:
                moveimg = np.ones((height, width, 3)) * 255

            drawOnCanvas(moveimg, tskins)

            cv2.imshow("Doodle", moveimg)
    

cv2.destroyAllWindows()
cv2.imwrite("MousePaint04.png", img)