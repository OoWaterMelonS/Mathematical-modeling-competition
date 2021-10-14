import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

# 导入数据
datafile = u'../data/V1_Molecular_Descriptor.xlsx'
data = pd.read_excel(datafile)
data_fea = data.iloc[:, 1:]  # 取数据中指标所在的列
data_fea = data_fea.fillna(0)  # 填补缺失值

# 标准化
data_mean = data_fea.mean()
data_std = data_fea.std()
data_fea = (data_fea - data_mean) / data_std

# 因子分析，并选取潜在因子的个数为10
FA = FactorAnalysis(n_components=25).fit_transform(data_fea.values)

# 潜在因子归一化
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
FA = min_max_scaler.fit_transform(FA)

# 绘制图像，观察潜在因子的分布情况
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.title('Factor Analysis Components')
plt.scatter(FA[:, 0], FA[:, 1])
plt.scatter(FA[:, 1], FA[:, 2])
plt.scatter(FA[:, 2], FA[:, 3])
plt.scatter(FA[:, 3], FA[:, 4])
plt.scatter(FA[:, 4], FA[:, 5])
plt.scatter(FA[:, 5], FA[:, 6])
plt.scatter(FA[:, 6], FA[:, 7])
plt.scatter(FA[:, 7], FA[:, 8])
plt.scatter(FA[:, 8], FA[:, 9])
plt.scatter(FA[:, 9], FA[:, 0])