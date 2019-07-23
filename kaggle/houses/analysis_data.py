# %%
from zipfile import ZipFile
import pandas as pd
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn


# %%
myzip = ZipFile(r"./kaggle/houses/data/house-prices-advanced-regression-techniques.zip")
df_train = pd.read_csv(myzip.open("train.csv"))
df_test = pd.read_csv(myzip.open("test.csv"))
myzip.close()
# df_train.columns
# Label SalePrice

# 将训练数据和测试数据连接
all_features = pd.concat((df_train.iloc[:,1:-1],df_test.iloc[:, 1:]))
print(all_features.shape)
# (2919, 79)

# %%
# 预处理数据集
# standardization
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / x.std()
)
# After standardization, mean becomes zero. we can fillna with 0
all_features[numeric_features].fillna(0, inplace=True)

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

# %%
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
    rmse = nd.sqrt(2*loss(clipped_preds.log() - labels.log()))
    return rmse.asscalar()

# Adam 算法相对于mini-Batch算法，对学习率相对不是那么敏感






#%%
