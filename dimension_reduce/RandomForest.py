import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

datafile = u'../data/V1_Molecular_Descriptor.xlsx'
data = pd.read_excel(datafile)
data_fea = data.iloc[:, 1:]  # 取数据中指标所在的列

model = RandomForestRegressor(random_state=1, max_depth=10)
# data_fea = data_fea.filln2a(0)  # 随机森林只接受数字输入，不接受空值、逻辑值、文字等类型
data_fea = pd.get_dummies(data_fea)
model.fit(data_fea, data.IC50_nM)
# 拿到
all_features = 729
# 根据特征的重要性绘制柱状图
# 拿到所有的列名字
labels = data_fea.columns
# 得到特征影响因子
importances = model.feature_importances_

indexs = np.argsort(importances)

res_features = 40
# 729 333  升序
tmp_indexs = np.flipud(indexs[-res_features-1:-1])


for i in range(res_features):
    print(str(importances[tmp_indexs[i]])+'-----'+str(labels[tmp_indexs[i]]))
    # # print(str(i+1)+'-'+str(labels[tmp_indexs[i]]))
    # print(str(labels[tmp_indexs[i]]))

labels[tmp_indexs].to_excel('../data/label.xlsx')

# todo   改成降序的情况
plt.title('Index selection')
plt.barh(range(len(tmp_indexs))[::-1], importances[tmp_indexs], color='blue', align='center')
plt.yticks(range(len(tmp_indexs))[::-1], [labels[i] for i in tmp_indexs])
plt.xlabel('Relative importance of indicators')
plt.show()

#  todo 归一化
# min_max_scaler = preprocessing.MinMaxScaler()
# influences_normalize = min_max_scaler.fit_transform(importances[tmp_indexs].reshape(-1, 1))
# # # 打印出归一化的结果
# for i in range(len(importances[tmp_indexs])):
#     print(influences_normalize[i])
