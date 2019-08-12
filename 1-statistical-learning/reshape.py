import matplotlib.pyplot as plt
from sklearn import datasets
digits = datasets.load_digits()
print(digits.images.shape)

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
plt.show()

input()

#reshape with -1 inferrs the other dimension - 3D tensor becomes 2D array
data = digits.images.reshape((digits.images.shape[0],-1))

print(data.shape)