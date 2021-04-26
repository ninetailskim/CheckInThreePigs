from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def addText( img, text, left, top, textColor=(0, 0, 0), textSize=50):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")

    draw.text((left+1, top+1), text, (0, 0, 0), font=fontStyle)
    draw.text((left, top), text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

img = np.zeros([50,50], dtype=np.uint8)
img = addText(img, "我", 0,0,textColor=(255,255,255))
cv2.imshow("t", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

lin = "我爱你,亲爱的姑娘"
print(len(lin))

lin = "吔屎啦你！"
print(len(lin))
