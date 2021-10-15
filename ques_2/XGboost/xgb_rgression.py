import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
from xgboost.sklearn import XGBClassifier
from sklearn import metrics

# 加载数据
datafile = u'../train_data/20_Molecular_Descriptor_train.xlsx'
X = pd.read_excel(datafile).iloc[:-24, :]

targetfile = u'../train_data/ERα_activity_train.xlsx'
y = pd.read_excel(datafile)['IC50_nM'].iloc[:-24, :]

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 定义超参数
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}
# 训练
dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 300
plst = list(params.items())
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
X_test = xgb.DMatrix(X_test)
ans = model.predict(X_test)
print(ans)

model.save_model('./model/xgb_regression.xgb')
