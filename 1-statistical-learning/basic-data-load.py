from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
print(data.shape)
print(data)