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

        if kernel =='min':
            self.kernel_function = lambda a,b:  np.sum(np.min(np.array([a,b]),axis=0))
        if kernel =='linear':
            self.kernel_function = lambda a,b : np.inner(a,b)
        if kernel =='rbf':
            self.kernel_function = lambda a,b : np.exp(- self.gamma * np.linalg.norm(a-b)**2)

    def gram_matrix(self, X):
        n_samples , n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        kernel_function = self.kernel_function
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i,j] = kernel_function(x_i, x_j)
        return K 


    def fit(self, X, y):
    	self.x_train = X

    	K = self.gram_matrix(X)
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
        	clf = KRR( lmb = self.lmb)
        	clf.fit(K,y_new)
        	clfs.append(clf)
        self.clfs = clfs
        self.labels = labels 

    def predict(self, X):
        y_pred = []
        y = np.zeros(shape=(X.shape[0], len(self.labels)))
        i = 0
        K_test = np.array([np.array([self.kernel_function(x,x2) for x2 in self.x_train])for x in X])
        for clf in self.clfs: 
            p = clf.predict(K_test)
            y[:,i] = p 
            i += 1
        for j in range(X.shape[0]):
        	y_pred.append(self.labels[np.argmax(y[j,:])])
        return y_pred 

  






