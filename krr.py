import numpy as np
import scipy
from numpy.linalg import norm


class KRR:
    def __init__(self, lmb = 0.1, kernel = 'linear', gamma = 0.01):
        self.lmb = lmb
        self.b = None
        self.alpha = None
        self.x_train = None
        self.gamma = gamma
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
        K = self.gram_matrix(X)
        K_arr = np.asarray(K, dtype=np.float)
        y_arr = np.asarray(y, dtype=np.float)
        self.x_train = X
        # dual solution 
        # (K + lambda I) alpha = y
        
        # Solve |K+lambdaI 1| |alpha| = |y|
        #       |    1     0| |  b  |   |0|
        n = K_arr.shape[0]
        A = np.empty((n+1, n+1), dtype=np.float)
        A[:n, :n] = K_arr + self.lmb * np.eye(n)
        A[n, :n], A[:n, n], A[n, n] = 1., 1., 0.
        g = np.linalg.solve(A, np.append(y_arr, 0))

        self.alpha, self.b = g[:-1], g[-1]
        return self 


    def predict(self, X):
        K_test = np.array([np.array([self.kernel_function(x,x2) for x2 in self.x_train])for x in X])
        print K_test.shape 
        Kt_arr = np.asarray(K_test, dtype=np.float)
        p = np.dot(self.alpha, Kt_arr.T) + self.b
        return p 