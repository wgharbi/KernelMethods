import numpy as np
import scipy
from numpy.linalg import norm


class KRR:
	def __init__(self, lmb = 0.1, kernel = 'linear'): 
		self.lmb = lmb
		self.b = None
		self.alpha = None 
		self.x_train = None 

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
    	self.x_train = None 
    	self.y_train = None 

    	n_samples = K_arr.shape[0]

        # dual solution 
        # (K + lambda I) alpha = y

        # Solve |K+lambdaI 1| |alpha| = |y|
        #       |    1     0| |  b  |   |0|
        # as in G. C. Cawley, N. L. C. Nicola and O. Chapelle.
        # Estimating Predictive Variances with Kernel Ridge 
        # Regression.

        A = np.empty((n_samples+1, n_samples+1), dtype=np.float)
        A[:n_samples, :n_samples] = K_arr + self._lmb * np.eye(n_samples)
        A[n, :n], A[:n, n], A[n, n] = 1., 1., 0.
        g = np.linalg.solve(A, np.append(y_arr, 0))
        self.alpha, self.b = g[:-1], g[-1]

    def predict(self, X): 
    	K_test = np.array([np.array([self.kernel_function(x,x2) for x2 in X])for x in self.x_train])
    	Kt_arr = np.asarray(K_test, dtype=np.float)
    	p = np.dot(self.alpha, Kt_arr.T) + self.b
    	return p 










