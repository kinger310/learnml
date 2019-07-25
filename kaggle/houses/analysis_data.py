# %% 1
from zipfile import ZipFile
import pandas as pd
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import d2lzh



# %% 2
myzip = ZipFile(r"D:/ml/learnml/kaggle/houses/data/house-prices-advanced-regression-techniques.zip")
df_train = pd.read_csv(myzip.open("train.csv"))
df_test = pd.read_csv(myzip.open("test.csv"))
myzip.close()
# df_train.columns
# Label SalePrice

# 将训练数据和测试数据连接
all_features = pd.concat((df_train.iloc[:,1:-1],df_test.iloc[:, 1:]))
print(all_features.shape)
# (2919, 79)

# %% 3
# 预处理数据集
# standardization
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
# After standardization, mean becomes zero. we can fillna with 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# scalar features to dummies
# dummy_na=True 将缺失值也作为一个合法的特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
# (2919, 331) fetures from 79 up to 331

# 通过values得到numpy格式的数据
n_train = df_train.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(df_train["SalePrice"].values).reshape((-1, 1))

# %% 4
# 训练模型
loss = gloss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net

# 对数均方根误差
# $$
# \sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(\log \left(y_{i}\right)-\log \left(\hat{y}_{i}\right)\right)^{2}}
# $$
def log_rmse(net, features, labels):
    # 将小于1的值置为1,便于log计算
    clipped_preds = nd.clip(net(features), 1, float("inf"))
    rmse = nd.sqrt(2*loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()

# Adam 算法相对于mini-Batch算法，对学习率相对不是那么敏感
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_loss, test_loss = [], []
    train_iter = gdata.DataLoader(
        gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True
    )
    # Adam

    trainer = gluon.Trainer(net.collect_params(), optimizer="adam",
                            optimizer_params={"learning_rate": learning_rate, "wd":weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)

        train_loss.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_loss.append(log_rmse(net, test_features, test_labels))

    return train_loss, test_loss

# %% 5
# k-fold cross-validation

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k,i,X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == 0:
            # """Plot x and log(y)."""
            d2lzh.semilogy(
                range(1, num_epochs+1), train_ls, "epoch", "rmse",
                range(1, num_epochs+1), valid_ls, legend=["train", "valid"]
            )
        print("fold %d, train rmse %f, valid rmse %f" % (i, train_ls[-1], valid_ls[-1]))

    return train_l_sum/k,valid_l_sum/k

# %% 6
# model selection
# 使用一组未经调优的超参数计算
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print("%d-fold validation: avg train rmse %f, avg valid rmse %f" %(k, train_l, valid_l))


# %% 7
# 预测并在kaggle上提交结果

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2lzh.semilogy(range(1, num_epochs+1), train_ls, "epoch", "rmse")
    print("train rmse %f" % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data["SalePrice"] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1)
    submission.to_csv("submission.csv", index=False)

train_and_pred(train_features, test_features, train_labels, df_test,
               num_epochs, lr, weight_decay, batch_size)