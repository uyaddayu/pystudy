import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
import numpy as np

# 　朴素贝叶斯分类器
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
# 这里我们有 4 个数据点，每个点有 4 个二分类特征。一共有两个类别：0 和 1。对于类别 0
# （第 1、3 个数据点），第一个特征有 2 个为零、0 个不为零，第二个特征有 1 个为零、1 个
# 不为零，以此类推。然后对类别 1 中的数据点计算相同的计数。
counts = {}
for label in np.unique(y):
    # 对每个类别进行遍历
    # 计算（求和）每个特征中1的个数
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))
