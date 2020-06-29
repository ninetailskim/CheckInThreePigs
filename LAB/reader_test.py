import argparse
import os
import numpy as np
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm


trainset = paddle.dataset.mnist.train()
batch_reader = paddle.batch(trainset, batch_size=1)

def proprecess(img):
    print(img)
    input()
    print(img[0])
    input()
    print(img[0].shape)
    input()
    img = np.array(img[0])
    print(img.shape)
    input()
    mu = np.mean(img)
    print(mu)
    sigma = np.std(img)
    print(sigma)
    res = (img - mu) / sigma
    print(res.shape)
    print(res.size)
    res = np.expand_dims(res, axis=0)
    print(res.shape)
    print(res.size)
    print(res.shape[0])
    input()
    return res

x_reader = paddle.reader.xmap_readers(proprecess, trainset, process_num=1, buffer_size=8192)

for i in x_reader():
    print(type(i))
    print(len(i))
    break