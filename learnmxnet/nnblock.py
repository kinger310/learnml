# %%
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

from mxnet import ndarray as F

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256,activation="relu")
        self.output = nn.Dense(10)
    
    def forward(self, x):
        return self.output(self.hidden(x))



# %%
class Model(nn.Block):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        # use name_scope to give child Blocks appropriate names.
        with self.name_scope():
            self.dense0 = nn.Dense(20)
            self.dense1 = nn.Dense(20)

    def forward(self, x):
        x = F.relu(self.dense0(x))
        return F.relu(self.dense1(x))

model = Model()
model.initialize(ctx=mx.cpu(0))
model(F.zeros((10, 10), ctx=mx.cpu(0)))


# %%

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation="relu"))
        net.add(nn.Dense(2))
    return net

x = nd.random.uniform(shape=(3,5))
import sys
try:
    net = get_net()
    net.initialize()
    net(x)
except RuntimeError as e:
    sys.stderr.write(str(e))