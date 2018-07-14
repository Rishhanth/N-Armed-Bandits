import numpy as np
from eps_greedy import epsilon_greedy
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

def main(epsilon,true_rew,iterations):
	n=10
	interact=epsilon_greedy(n,epsilon)
	a=0
	accum=0
	average_rew=[]
	for i in range(iterations):
		n=interact.pull_arm()
		reward=interaction(n,true_rew)
		interact.stochastic_avg_exp_reward(n,reward)
		#print(str(i+1)+'\t'+str(n+1)+'\t'+str(np.argmax(interact.expected_reward)+1))
		average_rew.append(interact.expected_reward[n])
	return average_rew

iterations=1000
bandits=2000
average_reward0=np.zeros(iterations)
average_reward0_01=np.zeros(iterations)
average_reward0_1=np.zeros(iterations)
for i in range(bandits):
	print(i)
	true_rew=generate_true_rew() 
	average_reward0=np.add(average_reward0,main(0,true_rew,iterations))
	average_reward0_01=np.add(average_reward0_01,main(0.01,true_rew,iterations))
	average_reward0_1=np.add(average_reward0_1,main(0.1,true_rew,iterations))
plt.title('Results of $\epsilon$ greedy approach')
eps0,=plt.plot(average_reward0/float(bandits),label='$\epsilon$ = 0')
eps0_01,=plt.plot(average_reward0_01/float(bandits),label='$\epsilon$ = 0.01')
eps0_1,=plt.plot(average_reward0_1/float(bandits),label='$\epsilon$ = 0.1')
plt.show()
