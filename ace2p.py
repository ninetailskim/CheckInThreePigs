import paddlehub as hub
import cv2
import os
import numpy as np
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ace2p = hub.Module(name='ace2p')

img = cv2.imread("1.jpg")

h, w, _ = img.shape
img = cv2.resize(img, (int(h/3), int(w/3)))

res = ace2p.segmentation([img], use_gpu=True)

print(res[0]['data'])

resint = res[0]['data']

resint[resint != 2] = 0
resint[resint == 2] = 1


resfloat = resint.astype(np.float32)
blurmask = cv2.blur(resfloat, (5,5))

blurmask1 = blurmask[:,:,np.newaxis]
blurmask3 = np.repeat(blurmask1, 3, axis=2)
timg = copy.deepcopy(img)
onlyhair = (timg * blurmask3).astype(np.uint8)

blur = cv2.GaussianBlur(onlyhair, (11, 11), 0)
canny = cv2.Canny(blur, 50, 80)
cv2.imshow('canny', canny)
cv2.waitKey()
cv2.destroyAllWindows()

h, w = canny.shape
print(canny.shape)
canny[resint == 0] = 255
# canny[canny == 0] = 150
# canny[canny == 255] = 150
canny = cv2.GaussianBlur(canny, (9, 9), 0)
cv2.imshow('canny', canny)
cv2.waitKey()
cv2.destroyAllWindows()

canny3 = canny[:,:,np.newaxis]

img[canny < 50] = [0,255,255]
img[canny > 220] = [0,0,0]
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()

'''

# ace2p
# hair 1
# other 0

hair origin
# 1 blur
# 2 canny
# canny * ace2p
0 = hair
1 = ske
'''