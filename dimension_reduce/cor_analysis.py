import numpy as np
import pandas as pd


f = open('../data/important_label.txt')

labels = f.readlines()
labels = [i.strip('\n') for i in labels]


datafile = u'../data/V2_Molecular_Descriptor.xlsx'
data = pd.read_excel(datafile)
data_fea = data.iloc[:, 1:]  # 取数据中指标所在的列

data_fea = np.array(data_fea)

# corr = data.corr(method='kendall')
# corr.to_excel('../data_corr/kendall_cor.xlsx', index=labels)

# corr = data.corr(method='spearman')
# corr.to_excel('../data_corr/spearman_cor.xlsx', index=labels)

corr = data.corr(method='pearson')
corr.to_excel('../data_corr/pearson_cor.xlsx', index=labels)

