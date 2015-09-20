import numpy as np
from scipy.optimize import minimize as mymin 
import pickle, math

class MyMaxEnt(object):
	'''
		Python Class for the Maximum Entropy Classifier
	'''
	def __init__(self, hist_list, feature_fn_list, tags=["PERSON","ORGANIZATION","GPE","MONEY","DATE","TIME"]):
		'''
			Initialises the Max Ent model by producing the feature vectors for the training data
		'''
		self.hist_list = hist_list
		self.tags = tags
		self.fvectors = self.create_dataset(feature_fn_list)
		self.init_model()  # initialise the model
		
		s = np.array([0]*10)
		temp = []
		for i in [self.fvectors[j].values() for j in self.fvectors.keys()]:
			for a in i:
				temp.extend(a) #shouldn't this be append?
		for i in temp:
			np.add(s,np.array(i))
		self.cum_f = s

	def init_model(self):
		'''
			Initialises the model parameter
		'''
		self.model = np.array([0]*10)

	def cost(self,model):
		'''
			Given the model, compute the cost 
		'''
		# we need to return negative of cost		
		L_of_v = sum([math.log(self.p_y_given_x(i,tag)) for i in self.fvectors.keys() for tag in self.tags])
		return -L_of_v

	def train(self):
		'''
			Train the classifier
		'''
		params = mymin(self.cost, self.model, method = 'L-BFGS-B', jac = gradient, options = {'disp' : True})
		self.model = params.x

	def p_y_given_x(self,h,tag):
		'''
			Take the history tuple and the required tag as the input and return the probability
		'''
		return float(math.exp(np.dot(self.model,self.fvectors[h][tag])))/sum([math.exp(np.dot(self.model,self.fvectors[h][tag])) for tag in self.tags])


	def classify(self,h):
		'''
			Performs the classification by determining the tag that maximizes the probability
		'''
		max_prob = 0
		best_tag = ""

		for tag in self.tags:
			prob = self.p_y_given_x(h,tag)
			if prob > max_prob:
				max_prob = prob
				best_tag = tag

		return best_tag

	def gradient(self, x):
		'''
			Maximizes the log-likelihood(Minimizes the negative)
		'''
		self.model = x
		temp = np.array([0]*10)
		for h in self.fvectors.keys():
			for tag in h.keys():
				np.add(temp, (self.fvectors[h][tag] * self.p_y_given_x(h, tag)))
		derivative = temp - self.cum_f
		return derivative
		
	def create_dataset(self, feature_fn_list):
		dataset = {k:{} for k in self.hist_list}
		for i in self.hist_list:
			for tag in self.tags:
				dataset[i][tag] = np.array([fun(i,tag) for fun in feature_fn_list])
		
		return dataset
		
	def load(self,model_file):
		'''
			Load the Max Ent model from the model file
		'''
		return pickle.load(open(model_file))

	def save(self,model_file):
		'''
			Save the model 
		'''
		pickle.dump(self,open((model_file),"w"))
