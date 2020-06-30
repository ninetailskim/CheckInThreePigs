import argparse
import os
import numpy as np
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm


trainset = paddle.dataset.mnist.train()
batch_reader = paddle.batch(trainset, batch_size=2)

def proprecess(img):
    print(img)
    input()
    img = np.array([i[0] for i in img], dtype="float32")
    print(img)
    input()
    mu = np.mean(img,axis=1)
    print(mu)
    input()
    sigma = np.std(img,axis=1)
    print(sigma)
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
    break