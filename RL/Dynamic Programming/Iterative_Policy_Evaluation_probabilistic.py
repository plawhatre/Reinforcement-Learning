import numpy as np
import WindyGridWorld 
from WindyGridWorld import action_space, standard_grid, print_values, print_policy 

if __name__=="__main__":
    # set transition probability and rewards
    transition_prob ={}
    rewards = {}
    g = standard_grid()
    for (s, a), v in g.prob.items():
        for s2, p  in v.items():
            transition_prob[(s,a,s2)] = p
            rewards[(s, a, s2)]  = g.rewards.get((s2),0)

    Policy ={
    (2,0) : {'U':0.5, 'R': 0.5},
    (1,0) : {'U': 1.0},
    (0,0) : {'R': 1.0},
    (0,1) : {'R': 1.0},
    (0,2) : {'R': 1.0},
    (1,2) : {'U': 1.0},
    (2,1) : {'R': 1.0},
    (2,2) : {'U': 1.0},
    (2,3) : {'L': 1.0},           
    }
    print("\nPolicy")
    print_policy(Policy, g)

    # Initilize V(s)
    V = {}
    for s in g.all_states():
        V[s] = 0

    #Discount Factor
    gamma = 0.9
    eps = 1e-5
    #repeat untill convergence
    itr = 0
    while True:
        biggest_change = 0
        for s in g.all_states():
            if not g.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in action_space:
                    for s2 in g.all_states():
                        action_prob = Policy[s].get(a, 0)
                        r = rewards.get((s,a,s2),0)
                        new_v += action_prob * transition_prob.get((s,a,s2),0)*(r + gamma*V[s2])
                V[s] = new_v
                biggest_change = max(biggest_change,np.abs(V[s]-old_v))
        print('Iteration=',itr,'Change=',biggest_change)
        print("Value function=\n\t")
        print_values(V, g)
        itr +=1
        if biggest_change <= eps:
            break

