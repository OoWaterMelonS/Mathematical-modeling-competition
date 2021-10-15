import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# 加载数据
datafile = u'../../train_data/Molecular_Descriptor_test.xlsx'
data = pd.read_excel(datafile).iloc[:, :]

linear_model = joblib.load('./model/linear_model.m')

# todo 代码还未测试
raws = data.raws

# 每一行有一个  结果
# todo  尝试直接把df给模型看能不能一次性预测
for i in range(1, raws):
    # 一行一行的进行  预测打印对应的结果
    res = linear_model.predict(np.array(data.iloc[i, :]).reshape(1, -1))
    print(res)
