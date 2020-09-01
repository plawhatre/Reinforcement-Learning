import numpy as np
import WindyGridWorld 
from WindyGridWorld import action_space, standard_grid_penalize, standard_grid, print_values, print_policy

def get_transition_prob_and_rewards(g):
	transition_prob ={}
	rewards = {}
	for (s, a), v in g.prob.items():
	        for s2, p  in v.items():
	            transition_prob[(s,a,s2)] = p
	            rewards[(s, a, s2)]  = g.rewards.get((s2),0)
	return transition_prob, rewards

def evalute_deterministic_policy(g, Policy):
	#Intilize V(s) = 0
	V = {}
	for s in g.all_states():
	    V[s] = 0
	# print("\nV(S)=\n\t",V)

	#Discount Factor
	gamma = 0.9
	eps = 1e-3
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
	    # print('Iteration=',itr,'Change=',biggest_change)
	    # print("Value function=\n\t")
	    # print_values(V, g)
	    itr +=1
	    if biggest_change <= eps:
	        break
	return V

if __name__ == '__main__':
	# g = standard_grid()
	g = standard_grid_penalize(-0.3)
	transition_prob, rewards = get_transition_prob_and_rewards(g)
	# print('Rewards=\n\t')
	# print_values(rewards, g)
	# Randomly select a policy
	Policy = {}
	for s in g.actions.keys():
		Policy[s] = np.random.choice(action_space)
	# print('Initial Policy\t')
	# print_policy(Policy, g)

	# Control 
	gamma = 0.9
	while True:
		# print_policy(Policy, g)
		V = evalute_deterministic_policy(g, Policy)
		# policy convergence
		is_policy_converged = True
		for s in g.actions.keys():
			old_a = Policy[s]
			new_a = None
			best_val = float('-inf')
			for a in action_space:
				v = 0
				for s2 in g.all_states():
					r = rewards.get((s,a, s2), 0)
					v += transition_prob.get((s, a, s2), 0)*(r +gamma*V[s2])
				if best_val< v:
					best_val = v
					new_a = a
			Policy[s] = new_a
			if new_a != old_a:
				is_policy_converged = False
		if is_policy_converged:
				break
	print('Values=')
	print_values(V, g)
	print('Policy=')
	print_policy(Policy, g)
