import numpy as np
import scipy
from numpy.linalg import norm


class KRR:
    def __init__(self, lmb = 0.1):
        self.lmb = lmb
        self.b = None
        self.alpha = None

    def fit(self, K, y):
        K_arr = np.asarray(K, dtype=np.float)
        y_arr = np.asarray(y, dtype=np.float)
       
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


    def predict(self, K): 
        Kt_arr = np.asarray(K, dtype=np.float)
        p = np.dot(self.alpha, Kt_arr.T) + self.b
        return p 