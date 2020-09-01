import numpy as np
import GridWorld 
from GridWorld import action_space, standard_grid, print_values, print_policy

if __name__=="__main__":
    transition_prob = {}
    rewards = {}

    g = standard_grid()
    # Set transition probability and reward
    for i in range(g.row):
        for j in range(g.col):
            s = (i,j)
            if not g.is_terminal(s):
                for a in action_space:
                    s2 = g.get_next_state(s, a)
                    transition_prob[(s, a, s2)] = 1
                    if s2 in g.rewards:
                        rewards[(s, a, s2)] = g.rewards[s2]

    print('Rewards=',rewards)
    # Fixed Policy
    Policy = {
    (2,0): 'U',
    (1,0) : 'U',
    (0,0) : 'R',
    (0,1) : 'R',
    (0,2) : 'R',
    (1,2) : 'U',
    (2,1) : 'R',
    (2,2) : 'U',
    (2,3) : 'L',
    }
    print("\nPolicy")
    print_policy(Policy, g)

    #Intilize V(s) = 0
    V = {}
    for s in g.all_states():
        V[s] = 0
    print("\nV(S)=\n\t",V)

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
                        action_prob = 1 if Policy[s] == a else 0
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