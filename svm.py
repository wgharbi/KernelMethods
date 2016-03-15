import numpy as np
import scipy
from numpy.linalg import norm
import cvxopt

class SVM:
    def __init__(self, C = 1., kernel = 'rbf', gamma = None):
        self.C = C
        if gamma == None:
            self.gamma = 0.01
        else:
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
        b = cvxopt.matrix(0.0)

        alphas = cvxopt.solvers.qp(P, Q, G, h, A, b)
        return np.ravel(alphas['x'])

    def fit(self, X, y): 
        #y must be a matrix of size (n_samples, 1)
        self.x_train = X
        self.classes = np.unique(y)
        print self.classes
        self.y_train = np.array([-1 if label == self.classes[0] else 1 for label in y])
        alphas = self.compute_multipliers(X, self.y_train)
        bias_list = []
        n_samples, n_features = X.shape
        K = self.gram_matrix(X)
        for j in range(n_samples): 
            result = np.sum([alphas[i]*self.y_train[i]*K[i,j] for i in range(n_samples)])
            b = self.y_train[j] - result
            bias_list.append(b)
        bias = np.mean(bias_list)
        self.bias = bias 
        self.alphas = alphas 

        return self


    def predict(self, X):
        prediction = []
        K_test = np.array([np.array([self.kernel_function(x,x2) for x2 in X])for x in self.x_train])
        for i,x in enumerate(X):
            result = self.bias
            for j in range(self.x_train.shape[0]):
                result+= self.alphas[j]*self.y_train[j]*K_test[j,i]
            if(np.sign(result) <0):
                prediction.append(self.classes[0])
            else:
                prediction.append(self.classes[1])

                
        return prediction





