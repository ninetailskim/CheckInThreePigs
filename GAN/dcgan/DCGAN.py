import argparse
import os
import numpy as np
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm
from paddle.fluid.param_attr import ParamAttr
import cv2

import shutil
if os.path.exists("images"):
    shutil.rmtree("images")
input()
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
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

        self.bn1 = BatchNorm(128,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0))
                            )
        self.conv1 = Conv2D(128, 128, 3, stride=1, padding=1,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0.0, 0.02))
                            )
        self.bn2 = BatchNorm(128,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0))
                            )
        self.conv2 = Conv2D(128, 64, 3, stride=1, padding=1,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0.0, 0.02))
                            )
        self.bn3 = BatchNorm(64,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0))
                            )
        self.conv3 = Conv2D(64, opt.channels, 3, stride=1, padding=1,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0.0, 0.02))
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

class DBlock(fluid.dygraph.Layer):
    def __init__(self, in_filter, out_filter, bn=True):
        super(DBlock, self).__init__()
        self.bn = bn
        self.conv = Conv2D( in_filter, 
                            out_filter, 3, stride=2, padding=1,  
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0.0, 0.02))
                            )
        self.bn = BatchNorm(out_filter,  
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
            out = self.bn(out)
        return out

class Discriminator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Discriminator, self).__init__(name_scope)
        self.model = []
        self.model.append(DBlock(opt.channels, 16, bn=False))
        self.model.append(DBlock(16, 32))
        self.model.append(DBlock(32, 64))
        self.model.append(DBlock(64, 128))
        self.adv_layer = Linear(128 * 4, 1)

    def forward(self, out):
        for layer_ in self.model:
            out = layer_(out)
            # print(out.shape)
        out = fluid.layers.reshape(out, shape=[out.shape[0], -1])
        out = self.adv_layer(out)
        out = fluid.layers.sigmoid(out)
        return out

trainset = paddle.dataset.mnist.train()
batch_reader = paddle.batch(trainset, batch_size=opt.batch_size)
shuffle = paddle.fluid.io.shuffle(batch_reader, 64)
def proprecess(img):
    img = np.array([i[0] for i in img], dtype="float32")
    # mu = np.expand_dims(np.mean(img, axis=1), axis=-1)
    # sigma = np.expand_dims(np.std(img, axis=1), axis=-1)
    # img = (img - mu) / sigma
    img = (img - 0.5) / 0.5
    img = img.reshape((-1,1,28,28))
    return img
x_reader = paddle.reader.xmap_readers(proprecess, shuffle, process_num=4, buffer_size=512)

adversarial_loss = fluid.dygraph.BCELoss()

def train(generator, discriminator):
    with fluid.dygraph.guard():
        generator.train()
        discriminator.train()

        total_steps = 935 * opt.n_epochs
        glr = fluid.dygraph.PolynomialDecay(0.002, total_steps, 0.0002)
        dlr = fluid.dygraph.PolynomialDecay(0.002, total_steps, 0.0002)
        # dlr = fluid.dygraph.PolynomialDecay(0.05, total_steps, 0.005)
        optimizer_G = fluid.optimizer.Adam(parameter_list=generator.parameters(), learning_rate=glr, beta1=opt.b1, beta2=opt.b2)
        optimizer_D = fluid.optimizer.Adam(parameter_list=discriminator.parameters(), learning_rate=dlr, beta1=opt.b1, beta2=opt.b2)

        for epoch in range(opt.n_epochs):
            for i, imgs in enumerate(x_reader()):
                valid = fluid.dygraph.to_variable(np.ones((imgs.shape[0], 1),dtype='float32'))
                valid.stop_gradient = True
                fake = fluid.dygraph.to_variable(np.zeros((imgs.shape[0], 1),dtype='float32'))
                fake.stop_gradient = True
                # print(imgs.shape)
                real_imgs = fluid.dygraph.to_variable(imgs)

                generator.clear_gradients()
                z = fluid.dygraph.to_variable(np.array(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)), dtype='float32'))

                gen_imgs = generator(z)

                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.minimize(g_loss)
                

                if i % 2 == 0:
                    discriminator.clear_gradients()
                    real_loss = adversarial_loss(discriminator(real_imgs), valid)
                    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

                    d_loss = (real_loss + fake_loss) / 2

                    d_loss.backward()
                    optimizer_D.minimize(d_loss)
                    

                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, 10, d_loss.numpy(), g_loss.numpy()))
                
                # print(gen_imgs.shape[0])
                timg = [gen_imgs[ti].numpy().transpose((2,1,0)) for ti in range(gen_imgs.shape[0])]
                # print(timg[0].shape)
                
                trow = []
                for ti in range(gen_imgs.shape[0] // 8):
                    # print(ti)
                    trow.append(np.concatenate(timg[ti*8:ti*8+8],axis=0))
                # print("concat 1")
                tre = np.concatenate(trow, axis=1)
                cv2.imshow("ss", tre)
                # print(tre.shape)
                cv2.waitKey(1)
                img = gen_imgs[0].numpy().transpose((2,1,0))
                img = cv2.resize(img, (280,280))
                cv2.imshow("ss2", img)
                cv2.waitKey(1)

                batches_done = epoch + i
                if batches_done % opt.sample_interval == 0:
                    for bi in range(opt.batch_size):
                        # print(gen_imgs.shape)
                        # print(gen_imgs[i].shape)
                        # print(np.squeeze(gen_imgs[i].numpy()).shape)
                        cv2.imwrite("images/%d_%d.png" % (batches_done, i), gen_imgs[bi].numpy().transpose((2,1,0)))      
            if epoch == opt.n_epochs - 1:
                input()
            fluid.save_dygraph(generator.state_dict(), './checkpoint/G_epoch{}'.format(epoch))
            fluid.save_dygraph(discriminator.state_dict(), './checkpoint/D_epoch{}'.format(epoch))

with fluid.dygraph.guard():
    generator = Generator("Generator")
    discriminator = Discriminator("Discriminator")
train(generator, discriminator)