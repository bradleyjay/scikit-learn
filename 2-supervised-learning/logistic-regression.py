import numpy as np 
from sklearn import datasets
from sklearn import linear_model

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

log = linear_model.LogisticRegression(solver='lbfgs', C=1e5, multi_class='multinomial')

log.fit(iris_X_train, iris_y_train)  

print(iris_X_test[])
print(iris_)
# plt.scatter(iris_X_test)
# plt.plot(iris_X_test, iris_y_test, linestyle='None', marker = 'o')
# plt.plot(iris_X_test, log.predict(iris_X_test))


# plt.scatter(iris_X_test, y, s=3) 
plt.show()