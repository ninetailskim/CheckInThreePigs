import argparse
import os
import numpy as np
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm

from paddlex.det import transforms

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_known_args()[0]
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

'''
cuda = True if torch.cuda.is_avaliable() else False
'''

class Gblock(fluid.dygraph.Layer):
    def __init__(self, in_feat, out_feat, is_normalize=True):
        super(Gblock, self). __init__()
        self.fc = Linear(in_feat, out_feat)
        self.bn = BatchNorm(out_feat)
        self.is_normalize = is_normalize

    def forward(self, x):
        out = self.fc(x)
        if self.is_normalize:
            out = self.bn(out)
        out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out         

class Generator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Generator, self).__init__(name_scope)
        print("create G")
        self.model = []
        self.model.append(self.add_sublayer('Gblock_0', Gblock(opt.latent_dim, 128, is_normalize=False)))
        self.model.append(self.add_sublayer('Gblock_1', Gblock(128, 256)))
        self.model.append(self.add_sublayer('Gblock_2', Gblock(256, 512)))
        self.model.append(self.add_sublayer('Gblock_3', Gblock(512, 1024)))
        self.Gblock_0 = Gblock(opt.latent_dim, 128, is_normalize=False)
        self.Gblock_1 = Gblock(128, 256)
        self.Gblock_2 = Gblock(256, 512)
        self.Gblock_3 = Gblock(512, 1024)
        self.fc = Linear(1024, int(np.prod(img_shape)))

    def forward(self, x):
        # out = self.model(x)
        # out = self.Gblock_0(x)
        # out = self.Gblock_1(out)
        # out = self.Gblock_2(out)
        # out = self.Gblock_3(out)
        for _layer in self.model:
            x = _layer(x)
        out = self.fc(x)
        out = fluid.layers.tanh(x=out)
        out = fluid.layers.reshape(x=out, shape=img_shape)
        return out

class Discriminator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Discriminator, self).__init__(name_scope)
        print("create D")
        self.fc1 = Linear(int(np.prod(img_shape)),512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 1)

    def forward(self, x):
        out = fluid.layers.reshape(x=x, shape=[x.shape[0], -1])
        out = self.fc1(out)
        out = fluid.layers.leaky_relu(x=out, alpha=0.2)
        out = self.fc2(out)
        out = fluid.layers.leaky_relu(x=out, alpha=0.2)
        out = self.fc3(out)
        out = fluid.layers.sigmoid(x=out)
        return out


trainset = paddle.dataset.mnist.train()
allCompose = transforms.Compose([transforms.Resize(target_size=opt.img_size), transforms.Normalize([0.5],[0.5])])
batch_reader = paddle.batch(trainset, batch_size=1)

def proprecess(img):
    img = np.array(img[0])
    mu = np.mean(img)
    sigma = np.std(img)
    res = (img - mu) / sigma
    return np.expand_dims(res, axis=0)

x_reader = paddle.reader.xmap_readers(proprecess, trainset, process_num=4, buffer_size=8192)

adversarial_loss = fluid.dygraph.BCELoss()
import cv2


def train(generator, discriminator):
    with fluid.dygraph.guard():
        
        
        optimizer_G = fluid.optimizer.Adam(parameter_list=generator.parameters(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
        optimizer_D = fluid.optimizer.Adam(parameter_list=discriminator.parameters(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
        for epoch in range(opt.n_epochs):
            for i, imgs in enumerate(x_reader()):
                # print(imgs)
                # print()
                generator.train()
                discriminator.eval()
                valid = fluid.dygraph.to_variable(np.ones((imgs.shape[0], 1),dtype='float32'))
                valid.detach()
                fake = fluid.dygraph.to_variable(np.zeros((imgs.shape[0], 1),dtype='float32'))
                fake.detach()

                real_imgs = fluid.dygraph.to_variable(imgs)

                generator.clear_gradients()
                z = fluid.dygraph.to_variable(np.array(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)), dtype='float32'))

                gen_imgs = generator(z)
                g_loss = adversarial_loss(discriminator(gen_imgs.detach()), valid)
                g_loss.backward()
                optimizer_G.minimize(g_loss)
                discriminator.train()
                discriminator.clear_gradients()
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.minimize(d_loss)

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, 10, d_loss.numpy(), g_loss.numpy()))

                batches_done = epoch * 10 + i
                if batches_done % opt.sample_interval == 0:
                    cv2.imwrite("images/%d.png" % batches_done, np.squeeze(gen_imgs.numpy()))      
        
#gpu_place = fluid.CUDAPlace(0)
with fluid.dygraph.guard():
    generator = Generator("Generator")
    discriminator = Discriminator("Discriminator")
train(generator, discriminator)