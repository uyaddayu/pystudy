import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 用于回归的线性模型
# 对于回归问题，线性模型预测的一般公式如下：
# ŷ = w[0] * x[0] + w[1] * x[1] + … + w[p] * x[p] + b
# 这里 x[0] 到 x[p] 表示单个数据点的特征（本例中特征个数为 p+1），w 和 b 是学习模型的
# 参数，ŷ 是模型的预测结果。对于单一特征的数据集，公式如下：
# ŷ = w[0] * x[0] + b
# 你可能还记得，这就是高中数学里的直线方程。这里 w[0] 是斜率，b 是 y 轴偏移。对于有
# 更多特征的数据集，w 包含沿每个特征坐标轴的斜率。或者，你也可以将预测的响应值看
# 作输入特征的加权求和，权重由 w 的元素给出（可以取负值）。
# 下列代码可以在一维 wave 数据集上学习参数 w[0] 和 b
mglearn.plots.plot_linear_regression_wave()
plt.show()
# 生成模型
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
# “斜率”参数（w，也叫作权重或系数）被保存在 coef_ 属性中，而偏移或截距（b）被保存在 intercept_ 属性中
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
# 训练集和测试集的性能
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# 我们来看一下 LinearRegression 在更
# 复杂的数据集上的表现，比如波士顿房价数据集。记住，这个数据集有 506 个样本和 105
# 个导出特征。首先，加载数据集并将其分为训练集和测试集。然后像前面一样构建线性回归模型
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
# 比较一下训练集和测试集的分数就可以发现，我们在训练集上的预测非常准确，但测试集上的 R2 要低很多
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
# 岭回归也是一种用于回归的线性模型，因此它的预测公式与普通最小二乘法相同。但在岭
# 回归中，对系数（w）的选择不仅要在训练数据上得到好的预测结果，而且还要拟合附加
# 约束。我们还希望系数尽量小。换句话说，w 的所有元素都应接近于 0。直观上来看，这
# 意味着每个特征对输出的影响应尽可能小（即斜率很小），同时仍给出很好的预测结果。
# 这种约束是所谓正则化（regularization）的一个例子。正则化是指对模型做显式约束，以
# 避免过拟合。岭回归用到的这种被称为 L2 正则化。
# 岭回归在 linear_model.Ridge 中实现。来看一下它对扩展的波士顿房价数据集的效果如何
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
# 简单性和训练集性能二者对于模型的重要程度可以由用户通过设置 alpha 参数来指定
# 增大 alpha 会使得系数更加趋向于 0，从而降低训练集性能，但可能会提高泛化性能
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))
# 减小 alpha 可以让系数受到的限制更小，对于非常小的 alpha 值，
# 系数几乎没有受到限制，我们得到一个与 LinearRegression 类似的模型
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
# 我们还可以查看 alpha 取不同值时模型的 coef_ 属性，从而更加定性地理解 alpha 参数是
# 如何改变模型的。更大的 alpha 表示约束更强的模型，所以我们预计大 alpha 对应的 coef_
# 元素比小 alpha 对应的 coef_ 元素要小
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()
# 还有一种方法可以用来理解正则化的影响，就是固定 alpha 值，但改变训练数据量
# 我们对波士顿房价数据集做二次抽样，并在数据量逐渐增加的子数据集上分
# 别对 LinearRegression 和 Ridge(alpha=1) 两个模型进行评估（将模型性能作为数据集大小
# 的函数进行绘图，这样的图像叫作学习曲线）
mglearn.plots.plot_ridge_n_samples()
# 岭回归和线性回归在波士顿房价数据集上的学习曲线
plt.show()
# 正如所预计的那样，无论是岭回归还是线性回归，所有数据集大小对应的训练分数都要高
# 于测试分数。由于岭回归是正则化的，因此它的训练分数要整体低于线性回归的训练分
# 数。但岭回归的测试分数要更高，特别是对较小的子数据集。如果少于 400 个数据点，线
# 性回归学不到任何内容。随着模型可用的数据越来越多，两个模型的性能都在提升，最终
# 线性回归的性能追上了岭回归。这里要记住的是，如果有足够多的训练数据，正则化变得
# 不那么重要，并且岭回归和线性回归将具有相同的性能（在这个例子中，二者相同恰好发
# 生在整个数据集的情况下，这只是一个巧合）。图 2-13 中还有一个有趣之处，就是线性回
# 归的训练性能在下降。如果添加更多数据，模型将更加难以过拟合或记住所有的数据。

#除了 Ridge，还有一种正则化的线性回归是 Lasso。与岭回归相同，使用 lasso 也是约束系
#数使其接近于 0，但用到的方法不同，叫作 L1 正则化。 L1 正则化的结果是，使用 lasso 时
#某些系数刚好为 0。这说明某些特征被模型完全忽略。这可以看作是一种自动化的特征选
#择。某些系数刚好为 0，这样模型更容易解释，也可以呈现模型最重要的特征。
#我们将 lasso 应用在扩展的波士顿房价数据集上
from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
#如你所见，Lasso 在训练集与测试集上的表现都很差。这表示存在欠拟合，我们发现模型
#只用到了 105 个特征中的 4 个。与 Ridge 类似，Lasso 也有一个正则化参数 alpha，可以控
#制系数趋向于 0 的强度。在上一个例子中，我们用的是默认值 alpha=1.0。为了降低欠拟
#合，我们尝试减小 alpha。这么做的同时，我们还需要增加 max_iter 的值（运行迭代的最大次数）

# 我们增大max_iter的值，否则模型会警告我们，说应该增大max_iter
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))
#但如果把 alpha 设得太小，那么就会消除正则化的效果，并出现过拟合，得到与LinearRegression 类似的结果
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

#样对不同模型的系数进行作图
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()