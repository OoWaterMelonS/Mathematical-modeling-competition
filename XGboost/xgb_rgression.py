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
datafile = u'../data/V2_Molecular_Descriptor.xlsx'
data = pd.read_excel(datafile)
# pandas数据框dataframe
X = data.iloc[:, 1:]
y = data.iloc[:, 1]

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
#训练
dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 300
plst = list(params.items())
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)
print(ans)

model.save_model('../model/xgb.xgb')
# 显示重要特征
# plot_importance(model)
# importance = model.get_score(importance_type='weight', fmap='')
#
# importance= sorted(importance.items(), key=lambda x: x[1],reverse=True)
# print(importance)
# plt.show()
