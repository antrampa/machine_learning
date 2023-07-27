import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "..")
from Preceptron.preceptron import Preceptron

# df = pd.read_csv(
#     "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
#     header=None,
# )
df = pd.read_csv("local_storage/iris.data", header=None)
df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

ppn = Preceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of updates")
plt.show()
