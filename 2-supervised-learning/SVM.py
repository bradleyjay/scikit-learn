from sklearn import svm
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# Ensure this random set of indicies exactly matches the tutorial by forcing the seed value
np.random.seed(0)
indicies = np.random.permutation(len(iris_X))

#use all but last 10 values to train
iris_X_train = iris_X[indicies[:-10]]
iris_y_train = iris_y[indicies[:-10]]

# use last 10 values to test
iris_X_test = iris_X[indicies[-10:]]
iris_y_test = iris_y[indicies[-10:]]

svc = svm.SVC(kernel='linear')
print(svc.fit(iris_X_train, iris_y_train))
