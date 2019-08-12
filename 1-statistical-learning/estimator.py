import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()

#reshape with -1 infers the other dimension
data = digits.images.reshape((digits.images.shape[0],-1))

print(data.shape)

estimator.fit(data)