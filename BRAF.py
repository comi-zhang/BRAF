import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, make_scorer, recall_score, precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 读取数据文件
data_file = "dataset/haberman.dat"
with open(data_file) as f:
    data = f.readlines()

# 从文件中获取属性和数据
attributes = []
data_list = []
for line in data:
    if line.startswith('@attribute'):
        attributes.append(line.strip().split()[1])
    elif not line.startswith('@'):
        data_list.append(line.strip().split(','))

# 将数据转换为DataFrame结构
df = pd.DataFrame(data_list, columns=attributes)

# 统计出少数类和多数类
class_counts = df['Class'].value_counts()
print("Class counts:\n", class_counts)

# 找出少数类
minority_class = class_counts.idxmin()
print("Minority class:", minority_class)

# 将数量比较少的类别单独存储在minority里面
minority = df[df['Class'] == minority_class]

# 将数据拆分为特征和标签
X = df.drop('Class', axis=1).astype(float)
y = df['Class']

minority_X = minority.drop('Class', axis=1).astype(float)
minority_y = minority['Class']

# 将数据集拆分为minority和majority
Tmaj = X[y != minority_class]
Tmin = X[y == minority_class]

# 建立KNN模型
k = 10
knn_model = KNeighborsClassifier(n_neighbors=k)

# 找到minority附近的k个最近邻
knn_model.fit(Tmin,minority_y)
distances, indices = knn_model.kneighbors(Tmaj,n_neighbors=k)

# 选择最近的k个邻居
nearest_indices = indices[:, :k]

# 将找到的maj最近邻和min添加到Tc中
unique_indices = set(nearest_indices.flatten()) - set(Tmin.index)
Tc = pd.concat([Tmin, Tmaj.iloc[list(unique_indices)]])

# 将Tc中的minority实例对应的标签设置为minority_y
minority_indices = Tc.index.isin(minority.index)
Tc.loc[minority_indices, 'Class'] = minority_y

# 处理Tc中的异常值Nan
Tc = Tc.dropna()

pd.set_option('display.max_rows', None)
print(Tc)
# BRAF的森林大小和偏向随机森林的树木占比
S = 100
p = 0.5

train_X,test_X,train_y,test_y = train_test_split(X,y)
# 基于完整数据集构建随机森林
rf1 = RandomForestClassifier(n_estimators=int(S*(1-p)), random_state=42)
rf1.fit(X, y)

# 基于关键数据集构建的带偏向的随机森林
rf2 = RandomForestClassifier(n_estimators=int(S*p), random_state=42)
rf2.fit(Tc.drop('Class', axis=1), Tc['Class'])

# 组合两个随机森林，生成BRAF随机森林
RF = RandomForestClassifier(n_estimators=S, random_state=42)
RF.estimators_ = rf1.estimators_ + rf2.estimators_

# 使用交叉验证评估BRAF随机森林的准确性
# 定义评判准则，输出结果
f1_scorer = make_scorer(f1_score, pos_label=minority_class)
scores = cross_val_score(RF, X, y, cv=10, scoring=f1_scorer)
print("Cross-validation scores:")
for score in scores:
    print(score,end=',')
print()
print("Mean F1 score:", scores.mean())
with open('result.txt', 'a') as f:
    f.write(f"{data_file}: Mean F1 score:{scores.mean()}\n")
gmean_scorer = make_scorer(geometric_mean_score, greater_is_better=True)
scores = cross_val_score(RF, X, y, cv=10, scoring=gmean_scorer)
print("Cross-validation scores:")
for score in scores:
    print(score,end=',')
print()
print("Mean gmean score:", scores.mean())
with open('result.txt', 'a') as f:
    f.write(f"{data_file}: Mean gmean score:{scores.mean()}\n")
