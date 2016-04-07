import pandas as pd
import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

path=""
X_train =pd.read_csv(path+"Xtr.csv", header=None)
Y =pd.read_csv(path+"Ytr.csv")
X_test =pd.read_csv(path+"Xte.csv", header=None)

y_train = Y["Prediction"].values
X_train = X_train.values
X_test = X_test.values

def center(X):
    image = X.reshape(28,28)
    X_center = 0.
    Y_center = 0.
    total = 0.
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            X_center += image[i,j] *i
            Y_center += image[i,j] *j
            total += image[i,j]
            
    X_center/= total
    Y_center/= total
    
    X_translate = X_center - 13
    Y_translate = Y_center - 13
    new_image = np.zeros((28,28))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]): 
            if(i+X_translate<0 or i+X_translate>27 or j+Y_translate<0 or j+Y_translate>27):
                new_image[i,j] = 0.
            else:
                new_image[i,j] = image[i+X_translate, j+Y_translate]
    return new_image
X_train = np.apply_along_axis(lambda x: center(x).flatten(),1,X_train)
X_test = np.apply_along_axis(lambda x: center(x).flatten(),1,X_test)

from HoG import hog_to_test
hist_test = hog_to_test(X_test, pix_per_cell = (4,4), cells_per_block = (3,3), orientation = 9)
hist = hog_to_test(X_train, pix_per_cell = (4,4), cells_per_block = (3,3), orientation = 9)
from multiclass_krr import multiclass_krr
krr = multiclass_krr(kernel = 'rbf', lmb = 0.00005, gamma = 0.01) 

krr.fit(hist, y_train)
prediction= krr.predict(hist_test)


def make_submission(predicted_label, name = 'submit.csv'):
    submit_d = d = {'Id' : pd.Series(np.arange(1,X_test.shape[0]+1).astype(int)),
                'Prediction' : pd.Series(predicted_label).astype(int)}
    submit = pd.DataFrame(submit_d)
    submit.to_csv(name,index=False)
    return submit

submit = make_submission(prediction)
