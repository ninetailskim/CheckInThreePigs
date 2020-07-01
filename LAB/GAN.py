import argparse
import os
import numpy as np
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm

from paddlex.det import transforms
import shutil

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
        out = fluid.layers.leaky_relu(x=out, alpha=0.2)
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
        out = fluid.layers.reshape(x=out, shape=[0,28,28,1])
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
batch_reader = paddle.batch(trainset, batch_size=opt.batch_size)
shuffle = paddle.fluid.io.shuffle(batch_reader, 64)
def proprecess(img):
    # img = np.array(img[0])
    # mu = np.mean(img)
    # sigma = np.std(img)
    # img = (img - mu) / sigma
    # img = np.expand_dims(img, axis=0)
    img = np.array([i[0] for i in img], dtype="float32")
    mu = np.expand_dims(np.mean(img, axis=1), axis=-1)
    sigma = np.expand_dims(np.std(img, axis=1), axis=-1)
    img = (img - mu) / sigma
    return img

x_reader = paddle.reader.xmap_readers(proprecess, shuffle, process_num=4, buffer_size=512)

adversarial_loss = fluid.dygraph.BCELoss()
import cv2


def train(generator, discriminator):
    with fluid.dygraph.guard():
        generator.train()
        discriminator.train()

        total_steps = 935 * opt.n_epochs
        lr = fluid.dygraph.PolynomialDecay(0.001, total_steps, 0.0001)

        optimizer_G = fluid.optimizer.Adam(parameter_list=generator.parameters(), learning_rate=lr, beta1=opt.b1, beta2=opt.b2)
        optimizer_D = fluid.optimizer.Adam(parameter_list=discriminator.parameters(), learning_rate=opt.lr / 10, beta1=opt.b1, beta2=opt.b2)
        for epoch in range(opt.n_epochs):
            for i, imgs in enumerate(x_reader()):
                # print(imgs)
                # print()
                
                # discriminator.eval()
                valid = fluid.dygraph.to_variable(np.ones((imgs.shape[0], 1),dtype='float32'))
                valid.stop_gradient = True
                fake = fluid.dygraph.to_variable(np.zeros((imgs.shape[0], 1),dtype='float32'))
                fake.stop_gradient = True

                real_imgs = fluid.dygraph.to_variable(imgs)
                
                z = fluid.dygraph.to_variable(np.array(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)), dtype='float32'))

                gen_imgs = generator(z)
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.minimize(g_loss)
                generator.clear_gradients()

                if i % 5 == 0:
                    # discriminator.train()
                    
                    real_loss = adversarial_loss(discriminator(real_imgs), valid)
                    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

                    d_loss = (real_loss + fake_loss) / 2

                    d_loss.backward()
                    optimizer_D.minimize(d_loss)
                    discriminator.clear_gradients()

                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, 10, d_loss.numpy(), g_loss.numpy()))

                cv2.imshow("ss", gen_imgs[0].numpy())
                cv2.waitKey(1)
                
                batches_done = epoch + i
                if batches_done % opt.sample_interval == 0:
                    for bi in range(opt.batch_size):
                        # print(gen_imgs.shape)
                        # print(gen_imgs[i].shape)
                        # print(np.squeeze(gen_imgs[i].numpy()).shape)
                        cv2.imwrite("images/%d_%d.png" % (batches_done, i), gen_imgs[bi].numpy())      
            
            fluid.save_dygraph(generator.state_dict(), './checkpoint/G_epoch{}'.format(epoch))
            fluid.save_dygraph(discriminator.state_dict(), './checkpoint/D_epoch{}'.format(epoch))
        
#gpu_place = fluid.CUDAPlace(0)
with fluid.dygraph.guard():
    generator = Generator("Generator")
    discriminator = Discriminator("Discriminator")
train(generator, discriminator)