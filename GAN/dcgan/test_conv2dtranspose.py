import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
    data = np.random.random((3, 5, 32, 32)).astype('float32')
    conv2DTranspose = fluid.dygraph.nn.Conv2DTranspose(num_channels=5,num_filters=2, filter_size=33)
    ret = conv2DTranspose(fluid.dygraph.base.to_variable(data))
    print(ret.shape)