import numpy as np  # 导入库
import pandas as pd

datafile = u'../data/V2_Molecular_Descriptor.xlsx'
data = pd.read_excel(datafile)
data_fea = data.iloc[:, 1:]  # 取数据中指标所在的列

correlation_matrix = np.corrcoef(data_fea, rowvar=0)  # 相关性分析
print(correlation_matrix.round(2))  # 打印输出相关性结果
