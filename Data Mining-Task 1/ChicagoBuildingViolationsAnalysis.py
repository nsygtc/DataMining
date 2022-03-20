import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier

# 读取数据
def readdata():
    path = './Chicago Building Violations/building-violations.csv'
    train = pd.read_csv(path)
    return train

# 数据摘要
def data_abstract(train):
    counts = train['VIOLATION STATUS'].value_counts()
    print("---VIOLATION STATUS---")
    print(counts)
    m = train.describe()
    print("---m---")
    print(m)
    n = train.isnull().sum()
    print("---n---")
    print(n)

# 数据可视化
def data_visual(train):
    plt.figure(figsize=(15, 8))
    sns.countplot(train['VIOLATION STATUS'])
    plt.title('VIOLATION STATUS')
    plt.figure(figsize=(10, 5))
    plt.boxplot(train['VIOLATION STATUS'].dropna())
    plt.title('VIOLATION STATUS')
    plt.show()

# 剔除缺失部分
def data_fill_1(train):
    plt.figure(figsize=(30, 10))
    plt.subplot(311)
    sns.countplot(train['Community Areas'])
    plt.title('Community Areas_old')
    plt.subplot(313)
    sns.countplot(train['Community Areas'].dropna())
    plt.title('Community Areas_new')
    plt.show()

# 用最高频率填补缺失值
def data_fill_2(train):
    plt.figure(figsize=(30, 10))
    plt.subplot(221)
    sns.countplot(train['Community Areas'])
    plt.title('Community Areas_old')
    plt.subplot(222)
    sns.countplot(train['Community Areas'].fillna(train['Community Areas'].value_counts().index[0]))
    plt.title('Community Areas_new')
    plt.figure(figsize=(10, 5))
    plt.subplot(223)
    plt.boxplot(train['Community Areas'].dropna())
    plt.title('Community Areas_old')
    plt.subplot(224)
    plt.boxplot(train['Community Areas'].fillna(train['Community Areas'].value_counts().index[0]))
    plt.title('Community Areas_new')
    plt.show()

# 通过属性的相关关系来填补缺失值
def data_fill_3(train):
    data = train[['Community Areas', 'LATITUDE', 'LONGITUDE']]
    print(data.shape)
    # data0 = data[['LATITUDE', 'LONGITUDE']].fillna(0)
    print(data['Community Areas'].notnull())
    test = data[data['Community Areas'].notnull()]
    test_x = test[['LATITUDE', 'LONGITUDE']].fillna(0)
    test_y = test[['Community Areas']]
    test_x = test_x.astype(float)
    test_y = test_y.astype(float)
    test_z = test[['Community Areas']]
    test_z = test_z.astype(float)
    train_t = data[data['Community Areas'].isnull()]
    train_x = train_t[['LATITUDE', 'LONGITUDE']].fillna(0).astype(float)
    train_y = train_t[['Community Areas']].astype(float)
    # print("test")
    test_z = test_z.values.reshape(-1)
    # print(test_x.shape)
    # print(test_y.shape)
    rfc = RandomForestClassifier()
    rfc.fit(test_x, test_z)
    pre = rfc.predict(train_x)
    # data[data['Community Areas'].isnull(), 'Community Areas'] = pre
    data.loc[(data['Community Areas'].isnull()), 'Community Areas'] = pre
    print(data.isnull().sum())
    plt.figure(figsize=(30, 10))
    plt.subplot(311)
    sns.countplot(x=train['Community Areas'])
    plt.title('Community Areas-Count_old')
    plt.subplot(313)
    sns.countplot(x=data['Community Areas'])
    plt.title('Community Areas-Count_new')
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.boxplot(train['Community Areas'].dropna())
    plt.title('boxplot_Community Areas_old')
    plt.subplot(133)
    plt.boxplot(data['Community Areas'])
    plt.title('boxplot_Community Areas_new')
    plt.show()


# 通过数据对象之间的相似性来填补缺失值
def data_fill_4(train):
    data_copy = train.copy(deep=True)
    imputer = KNNImputer(n_neighbors=3)
    data_copy['LATITUDE'].fillna(0)
    data_copy['LONGITUDE'].fillna(0)
    data_copy[['Community Areas', 'LATITUDE', 'LONGITUDE']] = imputer.fit_transform(
        data_copy[['Community Areas', 'LATITUDE', 'LONGITUDE']])
    print(data_copy.isnull().sum())
    plt.figure(figsize=(30, 10))
    plt.subplot(311)
    sns.countplot(x=train['Community Areas'])
    plt.title('Community Areas-Count_old')
    plt.subplot(313)
    sns.countplot(x=data_copy['Community Areas'])
    plt.title('Community Areas-Count_new')
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.boxplot(train['Community Areas'].dropna())
    plt.title('boxplot_Community Areas_old')
    plt.subplot(133)
    plt.boxplot(data_copy['Community Areas'])
    plt.title('boxplot_Community Areas_new')
    plt.show()

if __name__ == "__main__":
    train = readdata()
    data_abstract(train)
    data_visual(train)
    data_fill_1(train)
    data_fill_2(train)
    data_fill_3(train)
    data_fill_4(train)





