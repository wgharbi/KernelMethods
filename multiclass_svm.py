import cvxopt
import numpy as np
import scipy
from numpy.linalg import norm
import cvxopt
from svm import SVM
class multiclass_svm(object):
    def __init__(self, kernel = "rbf", C =1.):
        self.kernel = kernel
        self.C = C
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.trained_classifiers = [
            [SVM(C = self.C, kernel = self.kernel).fit(X[(y==label) | (y== label2)], y[(y==label)|(y== label2)])
                                                if label2>label else 0
                                                for label2 in self.classes] 
                                                for label in self.classes ]
        return self
        
    def predict(self, X):
        win_count = np.zeros((X.shape[0],len(self.classes)))
        for i, label1 in enumerate(self.classes):
            for j, label2 in enumerate(self.classes):
                if label2>label1:
                    y_test = self.trained_classifiers[i][j].predict(X)
                    for k, winner in enumerate(y_test):
                        if winner == label1:
                            win_count[k,i]+=1
                        else:
                            win_count[k,j]+=1
                            
        return [self.classes[np.argmax(row)] for row in win_count]
