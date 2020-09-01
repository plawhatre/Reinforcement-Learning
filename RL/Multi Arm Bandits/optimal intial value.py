import numpy as np
import matplotlib.pyplot as plt

class bandit:
    def __init__(self,p):
        self.p = p
        self.p_estimate = 5.0
        self.N = 1
        
    def pull(self):
        return np.random.rand()< self.p
    
    def update(self,x):
        self.N += 1
        self.p_estimate = ((self.N-1)*self.p_estimate+x)/self.N

def experiment(num_trials, bandits_prob):
    num_exploration = 0
    num_exploitation = 0
    num_optimal = 0
    reward = np.zeros(num_trials)
    bandits = [bandit(p) for p in bandits_prob]
    optimal_bandit = np.argmax([b.p_estimate for b in bandits])
    for i in range(num_trials):
        #select bandit
        j = np.argmax([b.p_estimate for b in bandits])
        if  j == optimal_bandit:
            num_optimal += 1 
        #pull the arm
        x = bandits[j].pull()
        reward[i] = x
        #reward
        bandits[j].update(x)
    #print details
    print("Number of exploration",num_exploration)
    print("Number of exploitation",num_exploitation)
    print("Number of times optimal bandit was selected",num_optimal)
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
    experiment(10000,[0.2,0.3,0.6])

