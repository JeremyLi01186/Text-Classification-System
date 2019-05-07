import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Simhei',size=13)

x = np.array([250,500,750,1000,2500,5000,7500,10000,50000,75000,100000])
y_bayes=np.array([86.00,86.00,83.00,83.00,82.00,82.00,83.00,83.00,85.00,85.00,85.00])
fig, ax = plt.subplots()
line1, = ax.plot(x, y_bayes, linewidth=2,
                 label='贝叶斯')
# line1.set_dashes(dashes)

ax.legend(loc='lower right')
ax.set_title("维度与分类器精确率的关系")
ax.set_xlabel("w维度")
ax.set_ylabel("准确率百分比")
plt.show()