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

if __name__ == '__main__':
	g = standard_grid()
	transition_prob, rewards = get_transition_prob_and_rewards(g)
	# Initialize V(s)
	V = {}
	for s in g.all_states():
		V[s] = 0
	itr = 0
	eps = 1e-3
	gamma = 0.9
	while True:
		biggest_change = 0
		for s in g.all_states():
			if not g.is_terminal(s):
				old_v = V[s]
				new_v = float('-inf')
				for a in action_space:
					v = 0
					for s2 in g.all_states():
						r = g.rewards.get(s2,0)
						v += transition_prob.get((s, a, s2), 0)*(r + gamma*V[s2])
					if new_v < v:
						new_v = v
				V[s] = new_v
				biggest_change = max(biggest_change, np.abs(V[s]-old_v))
		itr += 1
		if biggest_change < eps:
			break
	Policy = {}
	for s in g.actions.keys():
		best_a = None
		best_val = float('-inf')
		for a in action_space:
			v = 0
			for s2 in g.all_states():
				r = g.rewards.get((s, a, s2), 0)
				v += transition_prob.get((s, a, s2), 0)*(r + gamma*V[s2])
			if v > best_val:
				best_val = v
				best_a = a
		Policy[s] = best_a
	print('Values=')
	print_values(V, g)
	print('Policy=')
	print_policy(Policy, g)

