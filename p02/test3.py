import matplotlib
import numpy as np
import pandas as pd
import mglearn
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston

# 设置支持中文画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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


# k-NN 算法最简单的版本只考虑一个最近邻，也就是与我们想要预测的数据点最近的训练
# 数据点。预测结果就是这个训练数据点的已知输出
mglearn.plots.plot_knn_classification(n_neighbors=1)
# 显示出来
plt.show()
# 除了仅考虑最近邻，我还可以考虑任意个（k 个）邻居
mglearn.plots.plot_knn_classification(n_neighbors=3)
# 显示出来
plt.show()

# 如何通过 scikit-learn 来应用 k 近邻算法。首先，正如第 1 章所述，将数据分
# 为训练集和测试集，以便评估泛化性能
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 然后，导入类并将其实例化。这时可以设定参数，比如邻居的个数。这里我们将其设为 3
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)
# 现在，利用训练集对这个分类器进行拟合。对于 KNeighborsClassifier 来说就是保存数据
# 集，以便在预测时计算与邻居之间的距离
clf.fit(X_train, y_train)
# 调用 predict 方法来对测试数据进行预测。对于测试集中的每个数据点，都要计算它在训
# 练集的最近邻，然后找出其中出现次数最多的类别
print("Test set predictions: {}".format(clf.predict(X_test)))
# 评估模型的泛化能力好坏
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# 分析KNeighborsClassifier
# 对于二维数据集，我们还可以在 xy 平面上画出所有可能的测试点的预测结果。我们根据
# 平面中每个点所属的类别对平面进行着色。这样可以查看决策边界（decision boundary），
# 即算法对类别 0 和类别 1 的分界线
# 下列代码分别将 1 个、3 个和 9 个邻居三种情况的决策边界可视化
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit方法返回对象本身，所以我们可以将实例化和拟合放在一行代码中
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()

# 我们将在现实世界的乳腺癌数据集上进行研究。先将数据集分成训练集和测试集，然后用不同的邻居个
# 数对训练集和测试集的性能进行评估
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
# n_neighbors取值从1到10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # 构建模型
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 记录训练集精度
    training_accuracy.append(clf.score(X_train, y_train))
    # 记录泛化精度
    test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="训练集精度")
plt.plot(neighbors_settings, test_accuracy, label="测试集精度")
plt.ylabel("精度")
plt.xlabel("邻居数")
plt.legend()
# n_neighbors 为自变量，对比训练集精度和测试集精度
plt.show()

# k 近邻算法还可以用于回归。我们还是先从单一近邻开始，这次使用 wave 数据集。我们添
# 加了 3 个测试数据点，在 x 轴上用绿色五角星表示。利用单一邻居的预测结果就是最近邻的目标值
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()
# 同样，也可以用多个近邻进行回归。在使用多个近邻时，预测结果为这些邻居的平均值
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

# 用于回归的 k 近邻算法在 scikit-learn 的 KNeighborsRegressor 类中实现。其用法与KNeighborsClassifier 类似
from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)
# 将wave数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 模型实例化，并将邻居个数设为3
reg = KNeighborsRegressor(n_neighbors=3)
# 利用训练数据和训练目标值来拟合模型
reg.fit(X_train, y_train)

# 现在可以对测试集进行预测
print("Test set predictions:\n{}".format(reg.predict(X_test)))
# 我们还可以用 score 方法来评估模型，对于回归问题，这一方法返回的是 R2 分数。
# R2 分数也叫作决定系数，是回归模型预测的优度度量，位于 0 到 1 之间。R2 等于 1 对应完美预测，
# R2 等于 0 对应常数模型，即总是预测训练集响应（y_train）的平均值
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

# 为了便于绘图，我们创建一个由许多点组成的测试数据集
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# 创建1000个数据点，在-3和3之间均匀分布
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 利用1个、3个或9个邻居分别进行预测
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["模型预测", "Training data/target(训练集)",
                "Test data/target(测试集)"], loc="best")
plt.show()
# 从图中可以看出，仅使用单一邻居，训练集中的每个点都对预测结果有显著影响，预测结
# 果的图像经过所有数据点。这导致预测结果非常不稳定。考虑更多的邻居之后，预测结果
# 变得更加平滑，但对训练数据的拟合也不好
