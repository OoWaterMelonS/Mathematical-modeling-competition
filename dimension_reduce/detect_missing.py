import pandas as pd

datafile = u'../data/V1_Molecular_Descriptor.xlsx'
# datafile = u'../data/demo.xlsx'
data = pd.read_excel(datafile)
'''
所有行  去除第一列后的所有列   
因为 第一列中的为 因变量  IC50
'''
data_fea = data.iloc[1:, :]  # 取数据中指标所在的列

a = data_fea.isnull().sum() / len(data_fea) * 100  # 缺失值比例  换算为了百分比的值
# print('a='+a)
cols = data_fea.columns  # 列名
raw_name = data_fea.head()  # 列名
# print(raw_name)
raws = data_fea.shape[0]
raw = []
for i in range(0, raws):
    print('*'*20)
    print(a[i])
    if a[i] >= 50:  # 缺失值阈值为50%
        raw.append(raws[i])

print("缺失值低于阈值的特征共%d个；" % raws, "\n它们分别是：", raw)