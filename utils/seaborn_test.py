import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tips = pd.read_csv('tips.csv')
tips.head()

tips.corr()

sns.pairplot(tips)

#相关性图，和某一列的关系
sns.pairplot(tips ,hue ='sex', markers=["o", "s"])

# 相关性热力图
sns.heatmap(tips.corr())

# 分层相关性热力图
sns.clustermap(tips.corr())

g = sns.PairGrid(tips)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)