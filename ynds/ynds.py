import paddlehub as hub
import argparse
import cv2
import PIL 

class segUtils():
    def __init__(self):
        super(segUtils, self).__init__()
        self.module = hub.Module(name="deeplabv3p_xception65_humanseg")

    def do_seg(self, frame):
        res = self.module.segmentation(images=[frame], use_gpu=True)
        return res[0]['data']


def main(args):

    su = segUtils()
    leftimg = cv2.imread(args.leftimg)
    rightimg = cv2.imread(args.rightimg)
    
    leftp = su.do_seg(leftimg)
    rightp = su.do_seg(rightimg)

    with open("dialogue.txt", "r") as fp:
        lines = fp.readlines()
        for line in lines:
            speaker, words = line[0], line[1:]
            if speaker == 'A':
                
            else:




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--leftimg", type=str, required=True)
    parser.add_argument("--rightimg", type=str, required=True)
    parser.add_argument("--leftback", type=str, default="leftback.png")
    parser.add_argument("--rightback", type=str, default="rightback.png")
    parser.add_argument("--txtfile", type=str, required=True)
    args = parser.parse_args()
    main(args)