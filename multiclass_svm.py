import cvxopt
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
            self.kernel_function = lambda a,b : np.sum(np.minimum(a,b))
        if kernel =='linear':
            self.kernel_function = lambda a,b : np.inner(a,b)
        
        if kernel =='rbf':
            self.kernel_function = lambda a,b : np.exp(- self.gamma * np.linalg.norm( a-b)**2)
    def gram_matrix(self, X): 
        n_samples , n_features = X.shape 
        K = np.zeros((n_samples, n_samples))
        kernel_function = self.kernel_function
        for i, x_i in enumerate(X): 
            for j, x_j in enumerate(X): 
                K[i,j] = kernel_function(x_i, x_j)
        return K 

    def fit(self, X, y):
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.classes = np.unique(y)
        
        self.y_train = np.array([-1 if label == self.classes[0] else 1 for label in y])
        self.X_train = X
        y= self.y_train
        if (X.shape[0] != y.shape[0]):
            print "X and y don't have the same size :",X.shape, y.shape
            exit()
        
        #compute the kernel matrix
        K = self.gram_matrix(X)
        self.K = K
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.C)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))
        
        
        A = cvxopt.matrix(y, (1, n_samples),'d')
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = solution['x']
        self.b = np.mean([y[j] - np.sum([self.alpha[i]*y[i] * K[i,j] for i in range(n_samples)])
                         for j in range(n_samples)]).mean()
        return self
        
            
    def predict(self, X):
        prediction = []
        kernel= self.kernel_function
#        K_test = np.array([np.array([self.kernel_function(x,x2) for x2 in X])for x in self.X_train])
        for i,x in enumerate(X):
            result = self.b
            for j in range(self.X_train.shape[0]):
                result+= self.alpha[j]*self.y_train[j]*kernel(x_train[j],x)#K_test[j,i]
            if(np.sign(result) <0):
                prediction.append(self.classes[0])
            else:
                prediction.append(self.classes[1])
                
        return prediction        
            
    def predict(self, X):
        prediction = []
        for i,x in enumerate(X):
            result = self.b
            for j in range(self.X_train.shape[0]):
                result+= self.alpha[j]*self.y_train[j]*self.kernel_function(self.X_train[j],x)
            if(np.sign(result) <0):
                prediction.append(np.sign(self.classes[0]))
            else:
                prediction.append(np.sign(self.classes[1]))
                
        return prediction
#from svm import SVM
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
