import pandas as pd
import numpy as np

# datafile = u'../train_data/Molecular_Descriptor_train.xlsx'
datafile = u'../train_data/Molecular_Descriptor_test.xlsx'
# datafile = u'../train_data/Molecular_Descriptor_train.xlsx'
# datafile = u'../train_data/Molecular_Descriptor_train.xlsx'
# datafile = u'../train_data/Molecular_Descriptor_train.xlsx'
# datafile = u'../train_data/Molecular_Descriptor_train.xlsx'
orgin_data = pd.read_excel(datafile)
data = orgin_data.iloc[:, :]
cols = orgin_data.columns
labelfile = u'../train_data/50-to-20.xlsx'
labels = pd.read_excel(labelfile).iloc[0, :].tolist()
for i in range(len(cols)):
    if cols[i] not in labels:
        del data[cols[i]]
# train_data.to_excel('../train_data/20_Molecular_Descriptor_train.xlsx', index=False)
data.to_excel('../train_data/20_Molecular_Descriptor_test.xlsx', index=False)
