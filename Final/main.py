import numpy as np

from HoG import hog_to_test
from multiclass_svm import multiclass_svm
from multiclass_krr import multiclass_krr

X_train = pd.read_csv('Xtr.csv', header = None)
X_test  = pd.read_csv('Xte.csv')
y_train = pd.read_csv('Ytr.csv')

hist_train = hog_to_test(X_train, pix_per_cell = (4,4), cells_per_block = (3,3), orientation = 9)
hist_test = hog_to_test(X_train, pix_per_cell = (4,4), cells_per_block = (3,3), orientation = 9)

krr = multiclass_krr(kernel = 'rbf', lmb = TOBECOMPLETED, gamma = TOBECOMPLETED)
krr.fit(hist_train, y_train)
prediction = krr.predict(hist_test)