import numpy as np
import scipy
from numpy.linalg import norm
from krr import KRR 

class multiclass_krr(object):
    def __init__(self, kernel = 'linear', lmb = None, gamma = None):
        self.kernel = kernel
        self.clfs = None
        self.gamma = gamma
        if lmb == None: 
        	self.lmb = 0.01
        else:
        	self.lmb = lmb
        self.labels = None



    def fit(self, X, y): 
        labels = np.unique(y)
        labels_index = []
        for k in labels:
            labels_index.append([i for i, j in enumerate(y) if j == k])
        
        clfs = []
        
        for k, ktem in enumerate(labels_index):
        	print k
        	y_new = np.asarray(y.copy() ,dtype=np.float)
        	y_new[labels_index[k]] = 1.
        	y_new[[i for i, j in enumerate(y) if not i in labels_index[k]]] = -1. 
        	clf = KRR(kernel = self.kernel, gamma = self.gamma, lmb = self.lmb)
        	clf.fit(X,y_new)
        	clfs.append(clf)
        self.clfs = clfs
        self.labels = labels 

    def predict(self, X):
        y_pred = []
        y = np.zeros(shape=(X.shape[0], len(self.labels)))
        i = 0
        for clf in self.clfs: 
            p = clf.predict(X)
            y[:,i] = p 
            i += 1
        for j in range(X.shape[0]):
        	print j 
        	y_pred.append(self.labels[np.argmax(y[j,:])])
        return y_pred 

  






