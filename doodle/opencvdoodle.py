import cv2
import numpy as np

isMouseLBDown = False
circleColor = (0, 0, 0)
circleRadius = 5
lastPoint = (0, 0)

lines = []



def draw_circle(event, x, y, flags, param):

    global img
    global isMouseLBDown
    global color
    global lastPoint
    print(len(lines))
    if len(lines) > 0:
        print(lines[0])
    if event == cv2.EVENT_LBUTTONDOWN:


        isMouseLBDown = True
        cv2.circle(img, (x, y), int(circleRadius/2), circleColor, -1)
        lastPoint = (x, y)
        lines.append([(x,y)])
    elif event == cv2.EVENT_LBUTTONUP:

        isMouseLBDown = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if isMouseLBDown:
            cv2.line(img, pt1=lastPoint, pt2=(x,y), color=circleColor, thickness=circleRadius)
            lastPoint = (x, y)
            lines[-1].append((x,y))


def updateCircleColor(x):
    global circleColor
    global colorPreviewImg

    r = cv2.getTrackbarPos('Channel_Red', 'image')
    g = cv2.getTrackbarPos('Channel_Green', 'image')
    b = cv2.getTrackbarPos('Channel_Blue', 'image')

    circleColor = (b, g, r)
    colorPreviewImg[:] = circleColor


def updateCircleRadius(x):
    global circleRadius
    global radiusPreview

    circleRadius = cv2.getTrackbarPos('Circle_Radius', 'image')
    
    radiusPreview[:] = (255, 255, 255)
    cv2.circle(radiusPreview, center=(50, 50), radius=int(circleRadius / 2), color=(0,0,0), thickness=-1)


img = np.ones((512, 512, 3), np.uint8)
img[:] = (255, 255, 255)


colorPreviewImg = np.ones((100, 100, 3), np.uint8)
colorPreviewImg[:] = (0, 0, 0)

radiusPreview = np.ones((100,100,3), np.uint8)
radiusPreview[:] = (255, 255, 255)


cv2.namedWindow('image')

cv2.namedWindow('colorPreview')

cv2.namedWindow('radiusPreview')



cv2.createTrackbar('Channel_Red', 'image', 0, 255, updateCircleColor)
cv2.createTrackbar('Channel_Green', 'image', 0, 255, updateCircleColor)
cv2.createTrackbar('Channel_Blue', 'image', 0, 255, updateCircleColor)

cv2.createTrackbar('Channel_Radius', 'image',1, 20, updateCircleRadius)

cv2.setMouseCallback('image', draw_circle)

while(True):
    cv2.imshow('colorPreview', colorPreviewImg)
    cv2.imshow('radiusPreview', radiusPreview)
    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break
    

cv2.destroyAllWindows()
cv2.imwrite("MousePaint04.png", img)