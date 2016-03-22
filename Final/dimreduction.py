import numpy as np 
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

#data : the data matrix
#k the number of component to return
#return the new data and the variance that was maintained 
class pca():
	# Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
	# Rows of A correspond to observations (wines), columns to variables.
	# compute the mean
	# subtract the mean (along columns)
	# compute covariance matrix
	# compute eigenvalues and eigenvectors of covariance matrix
	# Sort eigenvalues (their indexes)
	# Sort eigenvectors according to eigenvalues
	# Project the data to the new space (k-D) and measure how much variance we kept
	def __init__(self, n_components = None):
		self.n_components = n_components

	def fit_transform(self, X): 
		M = X.mean(axis = 0) #compute the mean along columns
		C = X-M
		W = (C.T).dot(C) 
		eigval, eigvec = linalg.eig(W)
		
		idx = eigval.argsort()[::-1]
		eigval = eigval[idx]
		eigvec = eigvec[:,idx]
		k = self.n_components
		eigveck = eigvec[:, 0:k]
		X_transformed = np.dot(C, eigveck)
		self.components = eigveck
		self.mean =  X.mean(axis = 0)

		"""
		to compute how much of the data we have kept, we compute the sum of k eigenvalues that we have 
		kept over the sum of all eigen values. The variance information is contained in the eigenvalues of the covaruance matrix
		"""
		perc = sum(eigval[0:k])/sum(eigval)

		return  (X_transformed, perc)

	def transform(self, X): 
		
		C = X- self.mean
		X_transformed = np.dot(C, self.components)
		return X_transformed


class rbfpca():
	def __init__(self, n_components = None, gamma = None):
		self.n_components = n_components
		self.gamma = gamma 

	def fit_transform(self, X): 
		"""
		Implementation of a RBF kernel PCA.

		Arguments:
		X: A MxN dataset as NumPy array where the samples are stored as rows (M),
		and the attributes defined as columns (N).
		gamma: A free parameter (coefficient) for the RBF kernel.
		n_components: The number of components to be returned.

		Returns the k eigenvectors (alphas) that correspond to the k largest
		eigenvalues (lambdas).

		"""
		# Calculating the squared Euclidean distances for every pair of points
		# in the MxN dimensional dataset.
		sq_dists = pdist(X, 'sqeuclidean')

		# Converting the pairwise distances into a symmetric MxM matrix.
		mat_sq_dists = squareform(sq_dists)

		# Computing the MxM kernel matrix.
		gamma = self.gamma
		K = exp(-gamma * mat_sq_dists)

		# Centering the symmetric NxN kernel matrix.
		N = K.shape[0]
		one_n = np.ones((N,N)) / N
		K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
		eigvals, eigvecs = eigh(K_norm)
		# Obtaining the i eigenvectors (alphas) that corresponds to the i highest eigenvalues (lambdas).
		n_components = self.n_components
		alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
		lambdas = [eigvals[-i] for i in range(1,n_components+1)]
		self.lambdas = lambdas
		self.alphas = alphas
		self.X_fit = X
		return alphas

	def transform(self, X):
		X_fit = self.X_fit
		pairs_d = []
		for x in X:
			pair_d = [np.sum((x-row)**2) for row in X_fit]
			pairs_d.append(pair_d)
		pairs_dist = np.array(pairs_d)
		gamma = self.gamma
		k = np.exp(-gamma * pairs_dist)
		alphas = self.alphas 
		lambdas = self.lambdas
		return k.dot(alphas / lambdas)



    	 



