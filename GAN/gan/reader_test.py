import argparse
import os
import numpy as np
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm
import cv2
from paddlex.det import transforms

trainset = paddle.dataset.mnist.train()
allCompose = transforms.Compose([transforms.Resize(target_size=28), transforms.Normalize([0.5],[0.5])])
batch_reader = paddle.batch(trainset, batch_size=1)
shuffle = paddle.fluid.io.shuffle(batch_reader, 64)

def proprecess(img):
    img = np.array([np.expand_dims(i[0],axis=-1) for i in img], dtype="float32")
    print(img.shape)
    img = allCompose(img)
    print(type(img))
    print(img)
    return img


x_reader = paddle.reader.xmap_readers(proprecess, batch_reader, process_num=1, buffer_size=8192)

for i in x_reader():
    print(type(i))
    print(len(i))
    print(i.shape)
    for im in i:
        cv2.imshow("ss", im.reshape([28,28,1]))
        cv2.waitKey()
        print(im)
    break