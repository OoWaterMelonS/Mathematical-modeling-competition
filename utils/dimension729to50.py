import pandas as pd
import numpy as np

datafile = u'../data/V1_Molecular_Descriptor.xlsx'
orgin_data = pd.read_excel(datafile)
data = orgin_data.iloc[:, :]
cols = data.columns
# data_fea = data.iloc[:, 1:]  # 取数据中指标所在的列

f = open('../data/important_label.txt')

labels = f.readlines()
labels = [i.strip('\n') for i in labels]

for i in range(len(cols)):
    if cols[i] not in labels:
        del data[cols[i]]


data.to_excel('../data/V2_Molecular_Descriptor.xlsx', index=False)
