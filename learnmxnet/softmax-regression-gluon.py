
## 动手从零实现softmax
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn



batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 28*28
num_outputs = 10

# W = nd.random.normal(scale=.01, shape=(num_inputs, num_outputs))
# b = nd.zeros(num_outputs)
# W.attach_grad()
# b.attach_grad()

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=.01))


# define loss
loss = gloss.SoftmaxCrossEntropyLoss()


# 准确率
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# 训练模型
num_epochs = 5

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)

# 预测
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])

# Gluon提供的函数往往具有更好的数值稳定性