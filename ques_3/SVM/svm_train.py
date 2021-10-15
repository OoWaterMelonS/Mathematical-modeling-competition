import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# 加载数据
datafile = u'../../train_data/V2_Molecular_Descriptor.xlsx'
data = pd.read_excel(datafile).iloc[:, :]

targetfile = u'../../train_data/ADMET_train.xlsx'
target_Caco_2 = pd.read_excel(targetfile)['Caco-2']
target_CYP3A4 = pd.read_excel(targetfile)['CYP3A4']
target_HOB = pd.read_excel(targetfile)['HOB']
target_MN = pd.read_excel(targetfile)['MN']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(data, target_Caco_2, test_size=0.2, random_state=0)

# rbf_model = SVC(kernel='rbf')
# rbf_model.fit(X_train, y_train)  # 用训练数据拟合
# score = rbf_model.score(X_test, y_test)
# print(score)

linear_model = SVC(kernel='linear')
linear_model.fit(X_train, y_train)
score = linear_model.score(X_test, y_test)
print(score)

joblib.dump(linear_model, './model/svm_classify_Caco_2.m')
