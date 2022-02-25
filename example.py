from numpy import ndarray
from lm import lm
import seaborn as sns
import matplotlib.pyplot as plt

# use seaborn 'tips' dataset
tips = sns.load_dataset("tips")

# extract x and y
x: ndarray = tips.total_bill.to_numpy()
y: ndarray = tips.tip.to_numpy()

# linear regression by gradient descent
m, b = lm(x, y)

# show datapoints
sns.scatterplot(data=tips, x="total_bill", y="tip")

# draw a line from [0 f(0)] to [50 f(50)]
start: float = (m * 0) + b
end: float = (m * 50) + b
plt.plot([0, 50], [start, end], "r-", linewidth=2)

plt.show()
