import numpy as np
import matplotlib.pyplot as plt

class bandit:
    def __init__(self,p):
        self.p = p 
        self.p_estimate = 0
        self.N = 0
        
    def pull(self):
        return np.random.rand() < self.p
    
    def update(self, x):
        self.N +=1
        self.p_estimate = ((self.N-1)*self.p_estimate + x)/self.N

def ucb1(mean, N, nj):
    return mean + np.sqrt(2*np.log(N)/nj)

def experiment(num_trials,bandits_prob):
    reward = np.zeros(num_trials)
    bandits = [bandit(p) for p in bandits_prob]
    
    for i in range(len(bandits_prob)):
        x = bandits[i].pull()
        reward[i] = x 
        bandits[i].update(x)
    
    for i in range(1,num_trials):
        #select bandit
        j = np.argmax([ucb1(b.p_estimate,i, b.N) for b in bandits])
        #pull arm
        x = bandits[j].pull()
        reward[i] = x
        #update 
        bandits[j].update(x)
    #print details
    print("Total reward",reward.sum())
    print("_________________________\n")
    print("Estimation",[b.p_estimate for b in bandits])
    print("Ground Truth",bandits_prob)
    
    #plot reward
    plt.plot(np.cumsum(reward)/(np.arange(num_trials)+1))
    plt.plot(np.ones(num_trials)*np.max(bandits_prob))
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    experiment(20000,[0.2,0.5,0.8])

