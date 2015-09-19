import numpy as np
from scipy.optimize import minimize as mymin 

class MyMaxEnt(object):
	'''
	Python Class for the Maximum Entropy Classifier
	'''
	def __init__(self,hist_list,feature_fn_list):
		'''
		Initialises the Max Ent model by producing the feature vectors for the training data
		'''
		self.fvectors = self.create_dataset(hist_list)


	def init_model(self):
		'''
		Initialises the model parameter
		'''
		self.model = np.array([0]*10)

	def cost(self,model):
		'''
		Given the model, compute the cost 
		'''

	def train(self):
		'''
		Train the classifier
		'''
		params = mymin(self.cost, self.model, method = 'L-BFGS-B')
		#self.model = 

	def p_y_given_x(self,h,tag):
		'''
		Take the history tuple and the required tag as the input and return the probability
		'''

	def classify(self,h):
		'''
		Performs the classification by determining the tag that maximizes the probability
		'''

	def gradient(self):
		'''
		Maximizes the log-likelihood
		'''