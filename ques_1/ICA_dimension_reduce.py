import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# 导入数据
datafile = u'../train_data/V1_Molecular_Descriptor.xlsx'
data = pd.read_excel(datafile)
data_fea = data.iloc[:, 1:]  # 取数据中指标所在的列
data_fea = data_fea.fillna(0)  # 填补缺失值

# 标准化
data_mean = data_fea.mean()
data_std = data_fea.std()
data_fea = (data_fea - data_mean) / data_std

# 独立分量分析（ICA）
ICA = FastICA(n_components=3, random_state=20)
X = ICA.fit_transform(data_fea.values)

# 归一化
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# 绘制图像，观察成分独立情况
plt.figure(figsize=(12, 5))
plt.title('Factor Analysis Components')
plt.scatter(X[:, 0], ICA[:, 1])
plt.scatter(X[:, 1], ICA[:, 2])
plt.scatter(X[:, 2], ICA[:, 0])
plt.show()