import numpy as np
from ucb import UCB
import matplotlib.pyplot as plt

def sampler():
	return np.random.normal()

def interaction(n,true_rew):
	return true_rew[n]+sampler()

def generate_true_rew():
	true_rew=[]
	for i in range(10):
		true_rew.append(sampler())
	return true_rew

def main(true_rew,iterations):
	n=10
	interact=UCB(n)
	a=0
	accum=0
	average_rew=[]
	for i in range(iterations):
		n=interact.pull_arm(i)
		reward=interaction(n,true_rew)
		interact.stochastic_avg_exp_reward(n,reward)
		#print(str(i+1)+'\t'+str(n+1)+'\t'+str(np.argmax(interact.expected_reward)+1))
		average_rew.append(interact.expected_reward[n])
	return average_rew

iterations=1000
bandits=2000
average_reward0=np.zeros(iterations)
for i in range(bandits):
	true_rew=generate_true_rew() 
	average_reward0=np.add(average_reward0,main(true_rew,iterations))
plt.title("Results of UCB on 10-arm testbed")
plt.plot(average_reward0/float(bandits))
plt.show()
