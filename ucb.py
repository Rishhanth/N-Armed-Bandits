import numpy as np

class UCB():
	def __init__(self,no_of_arms,no_of_pulls=None,expected_reward=None):
		self.n=no_of_arms
		if (no_of_pulls and expected_reward) is None:
			self.pulls= np.zeros(self.n)
			self.expected_reward=np.zeros(self.n)			

		else:	 
			self.pulls=no_of_pulls
			self.expected_reward=expected_reward
			return
	
	def pull_arm(self,t):
		
		if t<self.n:
			return t
		else:
			return np.argmax(self.expected_reward+np.sqrt(2*np.log(t)*np.reciprocal(self.pulls.astype(float))))
	
	def stochastic_avg_exp_reward(self,arm_pulled,reward):
		self.pulls[arm_pulled]=self.pulls[arm_pulled]+1
		t=self.pulls[arm_pulled]
		estimate=self.expected_reward[arm_pulled]
		estimate=estimate + (reward-estimate)/float(t)
		self.expected_reward[arm_pulled]=estimate
		return

