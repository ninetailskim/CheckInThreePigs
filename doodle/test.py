import doodle
import paddlehub as hub
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class estUtil():
    def __init__(self):
        super(estUtil, self).__init__()
        self.module = hub.Module(name='human_pose_estimation_resnet50_mpii')

    def do_est(self, frame):
        res = self.module.keypoint_detection(images=[frame], use_gpu=True)
        return res[0]['data']

eu = estUtil()

img = cv2.imread('1.jpg')
res = eu.do_est(img)
# print(res)
# print(len(res))
ckeypoint = doodle.addcenterPoint(res)
# print(ckeypoint)
# print(len(ckeypoint))
ack, fas = doodle.complexres(res, doodle.FatherAndSon)
# print(ack)
# print(len(ack))
ack, fas = doodle.complexres(ack, fas)
print(ack)
print(len(ack))

# for key, value in fas.items():
#     print(key, "--", value)

for key in ack.keys():
    # print(key,'---',ack[key])
    cv2.circle(img,(int(ack[key][0]), int(ack[key][1])), 5, (0,0,255), -1)

nodes = doodle.toNodes(ack, fas)
# print("nodes:")
# for key, value in nodes.items():
#     print(key, "---", value.getPos())

nodes = doodle.connectNodes(nodes, fas)
# print("make father and son:")
# for key, value in nodes.items():
#     print(key, "---", value.parentName)
#     print(key, "---", fas[key])
#     if value.parentName != fas[key]:
#         print(False)


doodle.setInfo(nodes['centerpoint'])
doodle.travelTree(nodes['centerpoint'])
cv2.imshow("ss", img)
cv2.waitKey(0)
cv2.destroyAllWindows()