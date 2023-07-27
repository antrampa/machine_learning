import pandas as pd

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    header=None,
)
# df = pd.read_csv("local_storage/iris.data", header=None)
df.tail()

# https://github.com/python-poetry/install.python-poetry.org/issues/112
