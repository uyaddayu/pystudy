import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd

# 设置支持中文画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.title("包含 3 个类别的二维玩具数据集")
plt.show()

# 使用该数据对LinearSVC分类器进行训练
linear_svm = LinearSVC().fit(X, y)
print('模型斜率集：', linear_svm.coef_.shape)
print('模型截距集：', linear_svm.intercept_.shape)
# coef_ 的形状是 (3, 2)，说明 coef_ 每行包含三个类别之一的系数向量，每列
# 包含某个特征（这个数据集有 2 个特征）对应的系数值。现在 intercept_ 是一维数组，保
# 存每个类别的截距。
# 我们将这 3 个二类分类器给出的直线可视化
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
            'Line class 2'], loc=(1.01, 0.3))
plt.title("三个“一对其余”分类器学到的决策边界")
plt.show()
# 在这里，线条的颜色与各点的颜色是一致的。从图中可以很直观的看到这三个点是如何被分成三类的。
# 但是，这三条线交叉的地方，有一个空白的三角区，那这个区域属于哪个类别呢？
# 答案是分类方程结果最大的那个类别，即最接近的那条结对应的类别！
# 下面将展示整个二维空间是如何被分类的
mglearn.plots.plot_2d_classification(linear_svm, X, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0', 'Class 1', 'Class 2'], loc=(1.01, 0.3))
plt.title("三个“一对其余”分类器得到的多分类决策边界")
plt.show()
