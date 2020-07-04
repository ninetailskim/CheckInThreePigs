import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
    data = np.random.random((3, 5, 32, 32)).astype('float32')
    data = fluid.dygraph.to_variable(data)
    data = fluid.layers.image_resize(data, scale=2)
    print(data.shape)