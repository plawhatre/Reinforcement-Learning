import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_pdf

class bandit:
    def __init__(self,p):
        self.p = p
        self.alpha = 1
        self.beta = 1
        self.N = 0
        
    def sample(self):
        return np.random.beta(self.alpha,self.beta)
    
    def pull(self):
        return np.random.rand() < self.p
    
    def update(self,x):
        self.alpha += x
        self.beta += 1-x
        self.N += 1

def experiment(num_trials,bandit_prob):
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
        x = np.linspace(0,1,1000)
        y = beta_pdf.pdf(x, b.alpha, b.beta)
        estimates.append(x[np.argmax(y)])
        plt.plot(x,y)
    plt.show()
    print('estimates:peak',estimates)

if __name__ == "__main__":
    experiment(2000,[0.2,0.5,0.5])

