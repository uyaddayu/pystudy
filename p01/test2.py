import matplotlib
import numpy as np
import pandas as pd
import mglearn
from scipy import sparse
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.datasets import load_iris

# 设置支持中文画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 加载数据
iris_dataset = load_iris()

# 重新赋值成中文
iris_dataset['feature_names'] = ['花萼长度 (cm)', '花萼宽度 (cm)', '花瓣长度 (cm)', '花瓣宽度 (cm)']
# for name in iris_dataset['feature_names']:
#     print("{}".format(name))

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))
from sklearn.model_selection import train_test_split

# 为了确保多次运行同一函数能够得到相同的输出，我们利用 random_state 参数指定了随机数生成器的种子
# X_train 包含 75% 的行数据，X_test 包含剩下的 25%：
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_names中的字符串对数据列进行标记
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用DataFrame创建散点图矩阵，按y_train着色
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
                                 alpha=.8, cmap=mglearn.cm3)
# 显示出来
# plt.show()

# 第一个模型：k近邻算法
from sklearn.neighbors import KNeighborsClassifier

# n_neighbors是邻居的数目,设为 1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                     weights='uniform')

# 发现了一朵鸢尾花，花萼长 5cm 宽 2.9cm，花瓣长 1cm 宽0.2cm。
X_new = np.array([[5, 2.9, 1, 0.2]])
# 在一个 NumPy 数组,数组形状为样本数（1）乘以特征数（4）
print("X_new.shape: {}".format(X_new.shape))
# scikit-learn的输入数据必须是二维数组
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction]))
# 根据我们模型的预测，这朵新的鸢尾花属于类别 0，也就是说它属于 setosa 品种

# 使用测试集评估模型
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
# 通过计算精度（accuracy）来衡量模型的优劣，精度就是品种预测正确的花所占的比例
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
# 还可以使用 knn 对象的 score 方法来计算测试集的精度
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
