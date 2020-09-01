import numpy as np

class LinearModel:
	def __init__(self, input_dim, n_action):
		self.W = np.random.randn(input_dim, n_action)/ np.sqrt(n_action)
		self.b = np.zeros(n_action)
		#momentum terms
		self.vW = 0
		self.vb = 0
		self.losses = []

	def predict(self, X):
		assert len(X.shape)==2, 'X should be a 2 dimensional array'
		return X.dot(self.W) + self.b

	def sgd(self, X, Y, learning_rate=0.001, momentum=0.9):
		assert len(X.shape)==2, 'X should be a 2 dimensional array'

		num_values = np.prod(Y.shape)
		Y_hat = self.predict(X)
		#gradients
		gW = 2* X.T.dot(Y - Y_hat)/ num_values
		gb = 2* (Y - Y_hat).sum(axis= 0) / num_values
		#momentum
		self.vW = momentum*self.vW - learning_rate*gW   
		self.vb = momentum*self.vb - learning_rate*gb
		#Parameter Update
		self.W += self.vW
		self.b += self.vb
		#loss
		mse = np.mean((Y - Y_hat)**2)
		self.losses.append(mse)
	
	def load_params(self, filepath):
		npz = np.load(filepath)
		self.W = npz['W']
		self.b = npz['b']

	def save_params(self, filepath):
		np.savez(filepath, W=self.W, b=self.b)