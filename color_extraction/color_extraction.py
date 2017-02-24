#
from sklearn import cluster, datasets
#
#
# iris = datasets.load_iris()
# X_iris = iris.data
# y_iris = iris.target
#
# k_means = cluster.KMeans(n_clusters=8)
# k_means.fit(X_iris)
#
#
# print(iris.keys())
# print(iris.target_names)
# print(iris.feature_names)
# print(iris.data[:2])
# print(iris.target[:2])


# print(k_means.labels_[::10])
# print(y_iris[::10])

#data

from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np

#visualization
from bokeh.charts import Scatter, show
from bokeh.models import layouts
from bokeh.plotting import figure

iris = datasets.load_iris()
print(iris.keys())

X = pd.DataFrame(iris.data)
X.columns = iris.feature_names
X.columns = [str(x).replace(' ', '') for x in X.columns]
# print(X.columns)

y = pd.DataFrame(iris.target)
y.columns = ['y']


merge_data = X.join(y)
merge_data.y = iris.target_names[merge_data.y]
# print(merge_data[:4])


# p = Scatter(data=merge_data, x='sepallength(cm)', y='sepalwidth(cm)', color='Target'
#             , xlabel='length', ylabel='width', title='Sepal')
# show(p)

# p1 = Scatter(data=merge_data, x='petallength(cm)', y='petalwidth(cm)', color='Target'
#             , xlabel='length', ylabel='width', title='Petal')
# show(p1)


k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X)

merge_data = merge_data.join(pd.DataFrame(k_means.labels_,columns=['yHat']))
merge_data['yHat'].replace(0, 'setosa', inplace=True)
merge_data['yHat'].replace(1, 'versicolor', inplace=True)
merge_data['yHat'].replace(2, 'virginica', inplace=True)
# print(merge_data)
# print(pd.DataFrame(k_means.labels_,columns=['yHat']))


p = Scatter(data=merge_data, x='petallength(cm)', y='petalwidth(cm)', color='yHat'
            , xlabel='length', ylabel='width', title='Petal')
show(p)


































