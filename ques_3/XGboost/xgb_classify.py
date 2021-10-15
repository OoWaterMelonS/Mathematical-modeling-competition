from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# 加载数据
datafile = u'../train_data/20_Molecular_Descriptor_train.xlsx'
data = pd.read_excel(datafile).iloc[:, :]

targetfile = u'../train_data/ADMET_train.xlsx'
target_Caco_2 = pd.read_excel(targetfile)['Caco-2']
target_CYP3A4 = pd.read_excel(targetfile)['CYP3A4']
target_hERG = pd.read_excel(targetfile)['hERG']
target_HOB = pd.read_excel(targetfile)['HOB']
target_MN = pd.read_excel(targetfile)['MN']

X_train, X_test, y_train, y_test = train_test_split(data, target_Caco_2, test_size=0.2, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(data, target_Caco_2, test_size=0.2, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(data, target_Caco_2, test_size=0.2, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(data, target_Caco_2, test_size=0.2, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(data, target_Caco_2, test_size=0.2, random_state=1)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 2,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    # 'silent': 1,
    'eta': 0.001,
    'seed': 1000,
    'nthread': 4,
}

plst = list(params.items())

dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
X_test = xgb.DMatrix(X_test)
ans = model.predict(X_test)

# 计算准确率
cnt1 = 0
y_test = y_test.tolist()
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / len(y_test)))

# # 显示重要特征
# plot_importance(model)
# plt.show()

model.save_model('./model/xgb_classify_Caco_2.xgb')
