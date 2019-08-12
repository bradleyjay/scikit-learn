# split iris data in  train, test data
# don't test on training set

import numpy as np 
from sklearn import datasets

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

# Create, fit nearest-neighbor classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

print("Train the model:")
knn.fit(iris_X_train, iris_y_train) 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform')


print(knn.predict(iris_X_test))

print("Does this match our known Y labels? Let's see")
print(iris_y_test)