import argparse
import os
import numpy as np
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm

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

        self.l1 = Linear(opt.latent_dim, 128 * self.init_size ** 2,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0))
                            )

        self.bn1 = BatchNorm(128,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02))
                            )
        self.conv1 = Conv2D(128, 128, 3, stride=1, padding=1,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0))
                            )
        self.bn2 = BatchNorm(128,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)))
        self.conv2 = Conv2D(128, 64, 3, stride=1, padding=1,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0))
                            )
        self.bn3 = BatchNorm(64,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)))
        self.conv3 = Conv2D(64, opt.channels, 3, stride=1, padding=1,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0))
                            )

    def forward(self, z):
        out = self.l1(z)
        out = fluid.layers.reshape(out, shape=[out.shape[0], 128, self.init_size, self.init_size])
        out = self.bn1(out)
        out = fluid.layers.image_resize(out, scale=2)
        out = self.conv1(out)
        out = self.bn2(out)
        out = fluid.layers.leaky_relu(out, alpha=0.2)
        out = fluid.layers.image_resize(out, scale=2)
        out = self.conv2(out)
        out = self.bn3(out)
        out = fluid.layers.leaky_relu(out, alpha=0.2)
        out = self.conv3(out)
        out = fluid.layers.tanh(out)
        return out

class DBlock(fluid.dygraph.Layer)；
    def __init__(self, in_filter, out_filter, bn=True):
        self.bn = bn
        self.conv = Conv2D( in_filter, 
                            out_filter, 3, stride=2, padding=1,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0))
                            )
        
    def forward(self, out):
        out = self.conv(out)
        out = fluid.layers.leaky_relu(out)
        out = fluid.layers.dropout(out, dropout_prob=0.25)
        if self.bn:
            out = fluid.layers.batch_norm(out)
        return out

class Discriminator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Discriminator, self).__init__(name_scope)
        self.model = []
        self.model.append(DBlock(opt.channels, 16, bn=False))
        self.model.append(DBlock(16, 32))
        self.model.append(DBlock(32, 64))
        self.model.append(DBlock(64, 128))
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = Linear(128 * ds_size ** 2, 1,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02))
                            )

    def forward(self, out):
        for layer_ in self.model:
            out = layer_(out)
        out = fluid.layers.reshape(out, shape=[out.shape[0], -1])
        out = self.adv_layer(out)
        return out
        
adversarial_loss = fluid.dygraph.BCELoss()

with fluid.dygraph.guard():
    generator = Generator("Generator")
    discriminator = Discriminator("Discriminator")
