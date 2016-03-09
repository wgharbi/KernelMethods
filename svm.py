import numpy as np
import scipy
from numpy.linalg import norm
import cvxopt

class SVM:
    def __init__(self, C = 1.):
        self.C = C
        #self.kernel_function = lambda a,b : np.sum(np.minimum(a,b))
        self.kernel_function =lambda x,y : np.inner(x, y)

        
    def gram_matrix(self, X): 
        n_samples , n_features = X.shape 
        K = np.zeros((n_samples, n_samples))
        kernel_function = self.kernel_function
        for i, x_i in enumerate(X): 
            for j, x_j in enumerate(X): 
                K[i,j] = kernel_function(x_i, x_j)
        return K 

    def compute_multipliers(self, X, y): 
        #using cvxopt to solve the problem in the dual 
        #http://cvxopt.org/userguide/coneprog.html

        K = self.gram_matrix(X)
        n_samples , n_features = X.shape 
        Q = cvxopt.matrix(-1 * np.ones(n_samples))
        P = cvxopt.matrix(np.outer(y, y) * K)

        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

       
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.C)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples), 'd')
        print type(A)
        b = cvxopt.matrix(0.0)

        alphas = cvxopt.solvers.qp(P, Q, G, h, A, b)
        return np.ravel(alphas['x'])

    def fit(self, X, y): 
        #y must be a matrix of size (n_samples, 1)
        self.x_train = X
        self.y_train = y 
        alphas = self.compute_multipliers(X, y)
        bias_list = []
        n_samples, n_features = X.shape
        K = K = self.gram_matrix(X)
        for j in range(n_samples): 
            result = np.sum([alphas[i]*y[i]*K[i,j] for i in range(n_samples)])
            b = y[j] - result
            bias_list.append(b)
        bias = np.mean(bias_list)
        self.bias = bias 
        self.alphas = alphas 

        return self


    def predict(self, x): 
        #predict the label of a new point x 
        alphas = self.alphas 
        #TO DO: change SVM implementation to consider only support vectors in the predict 
        result= self.bias 
        k= self.kernel_function
        x_train = self.x_train
        n_samples_train, n_features_train = x_train.shape
        for j in range(n_samples_train): 
            result += alphas[j]*self.y_train[j]*k(x,x_train[j])
        
        return np.sign(result).item()






