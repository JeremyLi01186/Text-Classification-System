import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Simhei',size=13)

x = np.array([0.2,0.4,0.4,0.8,1.0])
x_bayes=np.array([0.79,0.80,0.80,0.81,0.71])
x_tree=np.array([0.87,0.69,0.69,0.65,0.45])
y_bayes=np.array([0.83,0.85,0.84,0.86,0.81])
y_tree=np.array([0.94,0.72,0.72,0.67,0.48])
fig, ax = plt.subplots()
line1, = ax.plot(x_bayes, y_bayes, linewidth=2,
                 label='贝叶斯')
# line1.set_dashes(dashes)

line2, = ax.plot(x_tree, y_tree,linewidth=2,
                 label='决策树')

ax.legend(loc='lower right')
ax.set_title("不同样本数量下精确率与召回率的关系")
ax.set_xlabel("召回率（Recall）")
ax.set_ylabel("准确率（Precision）")
plt.show()