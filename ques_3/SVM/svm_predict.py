import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import xgboost as xgb


linear_model = joblib.load('./model/linear_model.m')

datafile = u'../predict_data/20_Molecular_Descriptor_test.xlsx'
data = pd.read_excel(datafile).iloc[:, :]

X_predict = xgb.DMatrix(data)
ans = linear_model.predict(X_predict)
print(ans)

