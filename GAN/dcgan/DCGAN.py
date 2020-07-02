import argparse
import os
import numpy as np
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm, Conv2DTranspose

import shutil
if os.path.exists("images"):
    shutil.rmtree("images")
input()
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_known_args()[0]
print(opt)



class Generator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Generator, self).__init__(name_scope)
        self.init_size = opt.img_size // 4

        self.l1 = Linear(opt.latent_dim, 128 * self.init_size ** 2)
        self.model = []
        self.model.append(BatchNorm(128))
        self.model.append(Conv2DTranspose())

    def forward(self):
        pass