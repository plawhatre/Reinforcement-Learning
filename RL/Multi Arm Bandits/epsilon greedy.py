import numpy as np
import matplotlib.pyplot as plt

class bandit:
    def __init__(self,p):
        self.p = p 
        self.p_estimate = 0
        self.N = 0
    def pull(self):
        return np.random.rand() < self.p
    def update(self,x):
        self.N += 1
        self.p_estimate = ((self.N-1)*self.p_estimate + x)/self.N

def experiment(num_trials, eps, bandit_prob):
    
    bandits = [bandit(p) for p in bandit_prob]
    rewards = np.zeros(num_trials)
    num_exploration = 0
    num_exploitation = 0
    num_optimal = 0
    optimal_bandit = np.argmax([b.p for b in bandits])
    
    for i in range(num_trials):
        #select bandit for the experiment
        if np.random.rand() < eps:
            num_exploration += 1 
            j = np.random.randint(0,len(bandits))
        else:
            num_exploitation += 1 
            j = np.argmax([b.p_estimate for b in bandits])
        
        if j == optimal_bandit:
            num_optimal += 1 
        # Pull the arm
        x = bandits[j].pull()

        rewards[i] = x
        
        # update 
        bandits[j].update(x)
    
    #print details
    print("Number of exploration",num_exploration)
    print("Number of exploitation",num_exploitation)
    print("Number of times optimal bandit was selected",num_optimal)
    print("Total reward",rewards.sum())
    print("_________________________\n")
    print("Estimation",[b.p_estimate for b in bandits])
    print("Ground Truth",bandit_prob)
    
    #plot reward
    plt.plot(np.cumsum(rewards)/(np.arange(num_trials)+1))
    plt.plot(np.ones(num_trials)*np.max(bandit_prob))
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    experiment(100000,0.5,[0.2,0.3,0.8,0.9])

