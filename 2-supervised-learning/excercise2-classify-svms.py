from sklearn import svm
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data
y = iris.target


X = X[y != 0, :2]
y = y[y != 0]

np.random.seed(0)
indicies = np.random.permutation(len(X))

#use all but last 10% values to train. Find break point
b = int(.1*len(X))

X_train = X[indicies[:-b]]
y_train = y[indicies[:-b]]

# use last 10 values to test
X_test = X[indicies[-b:]]
y_test = y[indicies[-b:]]

print('Length of X:'  + str(len(X)))
print('Break!')
print(b)

print('Initial Sets:')

print('Train:')
print(X_train)
print(y_train)

print('Test:')
print(X_test)
print(y_test)

# old one
# svc = svm.SVC(kernel='linear')
# svc.fit(X_train, y_train)


# stolen goods

# fit the model
for kernel in ('linear', 'rbf', 'poly'):
    clf = svm.SVC(kernel=kernel, gamma=10, C=0.1)
    print('Model: ' + str(clf))
    clf.fit(X_train, y_train)

    plt.figure()
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
    print("Score: " + kernel)
    print(clf.score(X_test, y_test))
plt.show()
