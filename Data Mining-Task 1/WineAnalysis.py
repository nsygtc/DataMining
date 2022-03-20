import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute  import KNNImputer
from sklearn.linear_model import LinearRegression


# 读取数据
def readdata():
    path1 = './Wine Reviews/winemag-data_first150k.csv'
    path2 = './Wine Review/winemag-data-130k-v2.csv'
    train1 = pd.read_csv(path1)
    train2 = pd.read_csv(path2)
    return train1, train2

# 数据摘要
def data_abstract(train1, train2):
    counts1 = train1['country'].value_counts()
    print("---counts1---")
    print(counts1)
    counts2 = train2['country'].value_counts()
    print("---counts2---")
    print(counts2)
    m1 = train1.describe()
    print("---m1---")
    print(m1)
    n1 = train1.isnull().sum()
    print("---n1---")
    print(n1)
    m2 = train2.describe()
    print("---m2---")
    print(m2)
    n2 = train2.isnull().sum()
    print("---n2---")
    print(n2)

# 数据可视化
def data_visual(train1):
    plt.figure(figsize=(15, 8))
    plt.subplot(221)
    sns.countplot(train1.price)
    # sns.distplot(train1.price)
    plt.title('price')
    plt.subplot(222)
    # plt.figure(figsize=(15,5))
    sns.countplot(train1.points)
    plt.title('points')
    plt.subplot(223)
    plt.boxplot(train1.price.dropna())
    plt.title('price')
    plt.subplot(224)
    plt.boxplot(train1.points)
    plt.title('points')
    plt.show()

# 剔除缺失部分
def data_fill_1(train1):
    plt.figure(figsize=(15, 8))
    plt.subplot(311)
    sns.countplot(train1.price)
    plt.title('price_old')
    plt.subplot(313)
    sns.countplot(train1.price.dropna())
    plt.title('price_new')
    plt.show()

# 用最高频率填补缺失值
def data_fill_2(train1):
    plt.figure(figsize=(15, 8))
    plt.subplot(221)
    sns.countplot(train1.price)
    plt.title('price_old')
    plt.subplot(222)
    sns.countplot(train1.price.fillna(train1['price'].value_counts().index[0]))
    plt.title('price_new')
    plt.subplot(223)
    plt.boxplot(train1.price.dropna())
    plt.title('price_old')
    plt.subplot(224)
    plt.boxplot(train1.price.fillna(train1['price'].value_counts().index[0]))
    plt.title('price_new')
    plt.show()

# 通过属性的相关关系来填补缺失值
def data_fill_3(data1):
    data_pred = data1[np.isnan(data1['price'])]
    known_price = data1[data1.price.notnull()].values
    y = known_price[:, 0]  # price
    x = known_price[:, 1:]  # points
    line_reg = LinearRegression()
    line_reg.fit(x, y)
    data_pred['price'] = line_reg.predict(data_pred['points'].values.reshape(-1, 1))
    data1.loc[(data1.price.isnull()), 'price'] = data_pred['price']
    print(data1.shape)
    print(data1.isnull().sum())

# 通过数据对象之间的相似性来填补缺失值
def data_fill_null_4(data):
    data_copy = data.copy(deep=True)
    data_copy[['points', 'price']] = data_copy[['points', 'price']].replace(0, np.NaN)
    # null_index = data_copy.loc[data_copy['price'].isnull(), :].index
    imputer = KNNImputer(n_neighbors=3)
    data_copy[['points', 'price']] = imputer.fit_transform(data_copy[['points', 'price']])
    # print(data_copy.isnull().sum())
    # imputer = KNNImputer(n_neighbors=2)
    # imputer.fit_transform(data.price)
    # sns.countplot(data.price)


if __name__ == "__main__":
    train1, train2 = readdata()
    data_abstract(train1, train2)
    data_visual(train1)
    # data_visual(train2)
    data_fill_1(train1)
    # data_fill_1(train2)
    data_fill_2(train1)
    # data_fill_2(train2)
    data_fill_3(train1[['points', 'price']])
    # data_fill_3(train2[['points', 'price']])
    data_fill_null_4(train1[['points', 'price']])
    # data_fill_null_4(data=train2[['points', 'price']])

