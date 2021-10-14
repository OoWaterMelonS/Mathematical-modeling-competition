import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

datafile = u'../data/V2_Molecular_Descriptor.xlsx'
data = pd.read_excel(datafile)
# pandas数据框dataframe
X = data.iloc[:, :]

# 数据划分
model = xgb.Booster(model_file='../../model/xgb.xgb')

dtest = xgb.DMatrix(X)
ans = model.predict(dtest)
print(ans)