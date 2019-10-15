import matplotlib
import numpy as np
import pandas as pd
import mglearn
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston

# 生成数据集
X, y = mglearn.datasets.make_forge()
# 数据集绘图
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
# 显示出来
plt.show()
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
# 显示出来
plt.show()

# 包含在 scikit-learn 中的数据集通常被保存为 Bunch 对象,你可以用点操作符来访问对象的值（比如用 bunch.key 来代替 bunch['key']）。
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
# 这个数据集共包含 569 个数据点，每个数据点有 30 个特征
print("Shape of cancer data: {}".format(cancer.data.shape))
# 在 569 个数据点中，212 个被标记为恶性，357 个被标记为良性
print("Sample counts per class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
# 为了得到每个特征的语义说明，我们可以看一下 feature_names 属性
print("Feature names:\n{}".format(cancer.feature_names))

# 我们还会用到一个现实世界中的回归数据集，即波士顿房价数据集。与这个数据集相关的
# 任务是，利用犯罪率、是否邻近查尔斯河、公路可达性等信息，来预测 20 世纪 70 年代波
# 士顿地区房屋价格的中位数。这个数据集包含 506 个数据点和 13 个特征
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))
# 需要扩展这个数据集，输入特征不仅包括这 13 个测量结果，还包括这些特征
# 之间的乘积（也叫交互项）。换句话说，我们不仅将犯罪率和公路可达性作为特征，还将
# 犯罪率和公路可达性的乘积作为特征。像这样包含导出特征的方法叫作特征工程（feature engineering）
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
# 最初的 13 个特征加上这 13 个特征两两组合（有放回）得到的 91 个特征，一共有 104 个特征
