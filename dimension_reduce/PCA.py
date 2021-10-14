import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 导入数据
datafile = u'../data/V1_Molecular_Descriptor.xlsx'
data = pd.read_excel(datafile)
data_fea = data.iloc[:, 1:]  # 取数据中指标所在的列
data_fea = data_fea.fillna(0)  # 填补缺失值

# 标准化
data_mean = data_fea.mean()
data_std = data_fea.std()
data_fea = (data_fea - data_mean) / data_std
data_fea = data_fea.fillna(0)
# 选取主成分的个数为25
pca = PCA(n_components=30)
pca_result = pca.fit_transform(data_fea.values)

# 绘制图像，观察主成分对特征的解释程度
plt.bar(range(25), pca.explained_variance_ratio_, fc='pink', label='Single interpretation variance')
plt.plot(range(25), np.cumsum(pca.explained_variance_ratio_), color='blue', label='Cumulative Explained Variance')
plt.title("Component-wise and Cumulative Explained Variance")
plt.legend()
plt.show()