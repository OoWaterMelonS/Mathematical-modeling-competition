import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

datafile = u'./predict_data/Molecular_Descriptor_test.xlsx'
data = pd.read_excel(datafile)
# pandas数据框dataframe
X_test = data.iloc[:, :]

# datafile = u'../train_data/20_Molecular_Descriptor_train.xlsx'
# X_test = pd.read_excel(datafile).iloc[-24:, :]
#
# targetfile = u'../train_data/ERα_activity_train.xlsx'
# y = pd.read_excel(datafile)['IC50_nM'].iloc[-24:, :]


# 加载模型
model = xgb.Booster(model_file='./model/xgb_regression.xgb')
X_test = xgb.DMatrix(X_test)
ans = model.predict(X_test)

print(ans)
