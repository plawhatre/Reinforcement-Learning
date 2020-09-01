import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as normal_pdf

class bandit:
    def __init__(self,p):
        self.p = p
        self.m = 0
        self.lmbda = 1
        self.sum_x = 0
        self.tau = 1
        self.N = 0
        
    def sample(self):
        return self.m + np.random.randn()/np.sqrt(self.lmbda)
    
    def pull(self):
        return self.p + np.random.randn()/np.sqrt(self.tau)
    
    def update(self,x):
        self.lmbda += self.tau 
        self.sum_x += x
        self.m = (self.sum_x * self.tau)/self.lmbda        
        self.N += 1

def experiment(num_trials,bandit_prob):
    np.random.seed(1)
    bandits = [bandit(p) for p in bandit_prob]
    reward = np.zeros(num_trials)
    for i in range(num_trials):
        #select bandit_thompson sampling
        j = np.argmax([b.sample() for b in bandits])
        #pull arm
        x = bandits[j].pull()
        reward[i] = x
        #update
        bandits[j].update(x)
    #print details
    print("Total reward",reward.sum())
    print("_________________________\n")
    print("Ground Truth:probabilities",bandit_prob)
    
    #plot reward
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(np.cumsum(reward)/(np.arange(num_trials)+1))
    plt.plot(np.ones(num_trials)*np.max(bandit_prob))
    plt.xscale('log')
    #plot distribution
    plt.subplot(122)
    estimates = []
    for b in bandits:
        x = np.linspace(-100,100,1000)
        y = normal_pdf.pdf(x, b.m, b.lmbda)
        estimates.append(b.m)
        plt.plot(x,y)
    plt.show()
    print('estimates:peak',estimates)
    return bandits

if __name__ == "__main__":
    q = experiment(2000,[0.1,0.5,0.9])

