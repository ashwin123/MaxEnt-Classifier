import numpy as np
from scipy.optimize import minimize as mymin 
import pickle, math

class MyMaxEnt(object):
	'''
		Python Class for the Maximum Entropy Classifier
	'''
	def __init__(self,hist_list,feature_fn_list):
		'''
			Initialises the Max Ent model by producing the feature vectors for the training data
		'''
		self.fvectors = self.create_dataset(hist_list)
		self.tagset = ["PERSON","ORGANIZATION","GPE","MONEY","DATE","TIME"]

	def init_model(self):
		'''
			Initialises the model parameter
		'''
		self.model = np.array([0]*10)

	def cost(self,model):
		'''
			Given the model, compute the cost 
		'''
		# return L(v)

	def train(self):
		'''
			Train the classifier
		'''
		params = mymin(self.cost, self.model, method = 'L-BFGS-B', jac = gradient)
		#self.model = 

	def p_y_given_x(self,h,tag):
		'''
			Take the history tuple and the required tag as the input and return the probability
		'''
		return float(math.exp(np.dot(self.model,self.fvectors[(h,tag)])))/sum([math.exp(np.dot(self.model,self.fvectors[(h,tag)])) for tag in self.tagset])


	def classify(self,h):
		'''
			Performs the classification by determining the tag that maximizes the probability
		'''
		max_prob = 0
		best_tag = ""

		for tag in self.tagset:
			prob = self.p_y_given_x(h,tag)
			if prob > max_prob:
				max_prob = prob
				best_tag = tag

		return best_tag


	def gradient(self):
		'''
			Maximizes the log-likelihood(Minimizes the negative)
		'''
		
	def load(model_file):
		'''
			Load the Max Ent model from the model file
		'''
		self.model = pickle.load(open(model_file))

	def save(model_file):
		'''
			Save the model 
		'''
		pickle.dump(self.model,open((model_file),"w"))
