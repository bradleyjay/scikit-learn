'''

Model selection: choosing estimators and their parameters¶
https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

'''

# Score, and cross-validated scores¶

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn import datasets, svm
import matplotlib.pyplot as plt
digits = datasets.load_digits()

X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
c = svc.fit(X_digits[:-100], y_digits[:-100]
            ).score(X_digits[-100:], y_digits[-100:])
print(c)


X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)
# [0.934..., 0.956..., 0.939...]

# cross-validation generators
print('cross-validation generators')
X = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "c"]
k_fold = KFold(n_splits=5)
for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))


print([svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
       for train, test in k_fold.split(X_digits)])

# cross_val_score helper just...does this
n = cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
print('\n\n cross_val_score report: \n' + str(n))


# even simpler, cross_val_score helper just...does this
n = cross_val_score(svc, X_digits, y_digits, cv=k_fold,
                    n_jobs=-1, scoring='precision_macro')
print('\n\n cross_val_score report, precision_macro: \n' + str(n))

# Graph SVC score comparison


digits = datasets.load_digits()
X = digits.data
y = digits.target


C_s = np.logspace(-10, 0, 10)
score_array = []
plt.figure()

for c in C_s:
    svc = svm.SVC(kernel='linear', C=c)
    # defaults to 3 splits
    result = cross_val_score(svc, X, y, cv=3, n_jobs=-1)
    score_array.append(result)


for i in range(0, 3):
    # bracket because without anything, returns generator. Brackets
    # returns a list!

    plt.semilogx(C_s, [score[i] for score in score_array])

plt.show()


# print("\n Graphing attempt:")
# print(score_array)
