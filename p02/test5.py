import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 最常见的两种线性分类算法是 Logistic 回归（logistic regression）和线性支持向量机（linear
# support vector machine，线性 SVM），前者在 linear_model.LogisticRegression 中实现，
# 后者在 svm.LinearSVC（SVC 代表支持向量分类器）中实现
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import logistic
from sklearn.svm import LinearSVC
# 设置支持中文画图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 下面这行代码报警告：DeprecationWarning: Function make_blobs is deprecated; Please import make_blobs directly
# from scikit-learnwarnings.warn(msg, category=DeprecationWarning)
# 导入二分类数据
X, y = mglearn.datasets.make_forge()
# 本来想替换成下面这两行，结果报错 ValueError: cannot reshape array of size 3000000 into shape (1000,1000)
# from sklearn.datasets import make_blobs
# X, y = make_blobs()

# 建立一个幕布，两个绘图区
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
# 分别在两个绘图区上绘制两个模型的决策边界
for model, ax in zip([LinearSVC(max_iter=10000), LogisticRegression(solver='liblinear')], axes):
    # 在模型上训练数据
    clf = model.fit(X, y)
    # 绘制决策边界
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=.7)
    # 绘制二分类的训练数据
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    # 设置标题为模型名称
    ax.set_title("{}".format(clf.__class__.__name__))
    # 设置x坐标轴标签名称
    ax.set_xlabel("Feature 0")
    # 设置y坐标轴标签名称
    ax.set_ylabel("Feature 1")
# 在第一个绘图区上显示图例
axes[0].legend()
plt.show()  # 线性 SVM 和 Logistic 回归在 forge 数据集上的决策边界（均为默认参数）

# 对于 LogisticRegression 和 LinearSVC，决定正则化强度的权衡参数叫作 C。C 值越
# 大，对应的正则化越弱。换句话说，如果参数 C 值较大，那么 LogisticRegression 和
# LinearSVC 将尽可能将训练集拟合到最好，而如果 C 值较小，那么模型更强调使系数向量（w）接近于 0。
# 参数 C 的作用还有另一个有趣之处。较小的 C 值可以让算法尽量适应“大多数”数据点，
# 而较大的 C 值更强调每个数据点都分类正确的重要性。
mglearn.plots.plot_linear_svc_regularization()
# 不同 C 值的线性 SVM 在 forge 数据集上的决策边界
plt.show()

# 在乳腺癌数据集上详细分析 LogisticRegression
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
# 将数据分为训练集测试集
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# 使用模型训练数据
logreg = LogisticRegression(solver='liblinear').fit(X_train, y_train)
# 查看模型的评估值
print("logreg 训练集评估分数: {:.3f}".format(logreg.score(X_train, y_train)))
print("logreg 测试集评估分数: {:.3f}".format(logreg.score(X_test, y_test)))
# C=1 的默认值给出了相当好的性能，在训练集和测试集上都达到 95% 的精度。但由于训练
# 集和测试集的性能非常接近，所以模型很可能是欠拟合的。我们尝试增大 C 来拟合一个更灵活的模型
logreg100 = LogisticRegression(C=100, solver='liblinear').fit(X_train, y_train)
print("logreg100 训练集评估分数: {:.3f}".format(logreg100.score(X_train, y_train)))
print("logreg100 测试集评估分数: {:.3f}".format(logreg100.score(X_test, y_test)))
# 使用 C=100 可以得到更高的训练集精度，也得到了稍高的测试集精度，这也证实了我们的
# 直觉，即更复杂的模型应该性能更好。
# 我们还可以研究使用正则化更强的模型时会发生什么。设置 C=0.01
logreg001 = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
print("logreg001 训练集评估分数: {:.3f}".format(logreg001.score(X_train, y_train)))
print("logreg001 测试集评估分数: {:.3f}".format(logreg001.score(X_test, y_test)))  # 训练集和测试集的精度都比采用默认参数时更小
# 来看一下正则化参数 C 取三个不同的值时模型学到的系数

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.title("不同 C 值的 Logistic 回归在乳腺癌数据集上学到的系数")
plt.show()

# 如果想要一个可解释性更强的模型，使用 L1 正则化可能更好，因为它约束模型只使用少
# 数几个特征。下面是使用 L1 正则化的系数图像和分类精度
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1", solver='liblinear', max_iter=1000).fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)
plt.title("对于不同的 C 值，L1 惩罚的 Logistic 回归在乳腺癌数据集上学到的系数")
plt.show()
