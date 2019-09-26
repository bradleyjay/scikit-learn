'''



'''



from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from sklearn import datasets, svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')

Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),n_jobs=-1)

clf.fit(X_digits[:1000], y_digits[:1000])

clf.best_score_

clf.best_estimator_.C

# Prediction performance on test set is not as good as on train set
print(clf.score(X_digits[1000:], y_digits[1000:]))
