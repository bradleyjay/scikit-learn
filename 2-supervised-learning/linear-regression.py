import numpy as np 
from sklearn import datasets

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

print(regr)

print(regr.coef_)

print('Mean square error')

print(np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2))

print('Variance Score: 1 is perfect, 0 is no linear relationship')

print(regr.score(diabetes_X_test, diabetes_y_test))
