import argparse
import os
import numpy as np
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm
import cv2

trainset = paddle.dataset.mnist.train()
batch_reader = paddle.batch(trainset, batch_size=2)

def proprecess(img):
    print(img)
    input()
    img = np.array([i[0] for i in img], dtype="float32")
    print(img)
    print(img.shape)
    input()
    mu = np.mean(img,axis=1)
    print(mu)
    print(mu.shape)
    input()
    mu = np.expand_dims(mu, axis=-1)
    print(mu)
    print(mu.shape)
    input()
    sigma = np.std(img,axis=1)
    print(sigma)
    print(sigma.shape)
    input()
    sigma = np.expand_dims(sigma, axis=-1)
    print(sigma)
    print(sigma.shape)
    input()
    res = (img - mu) / sigma
    # res = np.expand_dims(res, axis=0)
    print(res.shape)
    input()
    return res

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