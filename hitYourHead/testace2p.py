import paddlehub as hub
import cv2
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class segUtils:
    def __init__(self):
        super(segUtils, self).__init__()
        self.ace2p = hub.Module(name='ace2p')

    def getMask(self, frame):
        res = self.ace2p.segmentation([frame], use_gpu=True)
        return res[0]['data']


img = cv2.imread("1.jpg")
su = segUtils()
res = su.getMask(img)
for i in range(19):
    print(i)
    t = np.zeros_like(img)
    t[res == i] = 1
    timg = img * t
    cv2.imshow("test", timg.astype(np.uint8))
    cv2.waitKey(0)
cv2.destroyAllWindows()