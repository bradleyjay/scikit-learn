from sklearn import datasets, neighbors, linear_model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

digits = datasets.load_digits()
X_digits = digits.data / digits.data.max()
y_digits = digits.target

np.random.seed(0)
indices = np.random.permutation(len(X_digits))

split = int(.9 * len(X_digits))

digits_X_train = X_digits[indices[:split]]
digits_y_train = y_digits[indices[:split]]
digits_X_test = X_digits[indices[split:]]
digits_y_test = y_digits[indices[split:]]

# # linear regression
regr = linear_model.LinearRegression()
regr.fit(digits_X_train, digits_y_train)
print(regr.score(digits_X_test, digits_y_test))

# nearest log
log = linear_model.LogisticRegression(solver='lbfgs', C=1e5, multi_class='multinomial',  max_iter=1000)
log.fit(digits_X_train, digits_y_train) 
print(log.score(digits_X_test, digits_y_test))

# nearest Kneighbors
knn = KNeighborsClassifier()
knn.fit(digits_X_train, digits_y_train) 
print(knn.score(digits_X_test, digits_y_test))


import matplotlib.pyplot as plt 
plt.scatter(digits_X_train[0,:],digits_X_train[1,:]) 
plt.show()