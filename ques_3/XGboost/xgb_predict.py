import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# 应该有五个模型 针对不同的  ADMET
model = xgb.Booster(model_file='./model/xgb_classify_Caco_2.xgb')

# 加载数据
datafile = u'../predict_data/20_Molecular_Descriptor_test.xlsx'
data = pd.read_excel(datafile).iloc[:, :]


X_predict = xgb.DMatrix(data)
ans = model.predict(X_predict)
print(ans)