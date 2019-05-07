import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Simhei',size=13)

x = np.array([700,1400,2100,2800,3500])
y_bayes=np.array([83.00,86.00,84.00,85.00,83.00])
y_tree=np.array([48.00,67.00,72.00,72.00,94.00])
fig, ax = plt.subplots()
line1, = ax.plot(x, y_bayes, linewidth=2,
                 label='贝叶斯')
# line1.set_dashes(dashes)

line2, = ax.plot(x, y_tree,linewidth=2,
                 label='决策树')

ax.legend(loc='lower right')
ax.set_title("贝叶斯与决策树文本分类准确率对比")
ax.set_xlabel("样本数量")
ax.set_ylabel("准确率百分比")
plt.show()