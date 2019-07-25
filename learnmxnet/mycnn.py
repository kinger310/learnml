import mxnet as mx
from mxnet import nd

# Input data should be 4D in batch-num_filter-y-x
data = nd.arange(9).reshape((1,1,3,3))
w = nd.arange(4).reshape((1,1,2,2))
b = nd.array([1])
out1 = nd.Convolution(data, w,b, kernel=w.shape[2:],num_filter=w.shape[1])

#
w = nd.arange(8).reshape((1,2,2,2))
b = nd.array([1])
data = nd.arange(18).reshape((1,2,3,3))
out2 = nd.Convolution(data, w,b, kernel=w.shape[2:],num_filter=w.shape[0])


from mxnet.gluon import nn

# 定义一个函数来计算卷积层
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    X = X.reshape((1,1)+X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2D(1, kernel_size=3, padding=1)

X = nd.random.uniform(shape=(8,8))
print(comp_conv2d(conv2d,X).shape)

# 5*3的卷积核
conv2d = nn.Conv2D(1, kernel_size=(5,3))
comp_conv2d(conv2d, X)