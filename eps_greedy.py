import numpy as np

class epsilon_greedy():
	def __init__(self,no_of_arms,epsilon,no_of_pulls=None,expected_reward=None):
		self.n=no_of_arms
		self.epsilon=epsilon
		if (no_of_pulls and expected_reward) is None:
			self.pulls= np.zeros(self.n)
			self.expected_reward=0.01*np.random.rand(self.n)
		else:	 
			self.pulls=no_of_pulls
			self.expected_reward=expected_reward
			return
	
	def pull_arm(self):
		if np.random.random() < self.epsilon:
			return np.random.randint(self.n)
		else:
			return np.argmax(self.expected_reward)
	
	def stochastic_avg_exp_reward(self,arm_pulled,reward):
		self.pulls[arm_pulled]=self.pulls[arm_pulled]+1
		t=self.pulls[arm_pulled]
		estimate=self.expected_reward[arm_pulled]
		estimate=estimate + (reward-estimate)/float(t)
		self.expected_reward[arm_pulled]=estimate
		return

		

