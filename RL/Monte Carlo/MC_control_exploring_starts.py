import numpy as np
import matplotlib.pyplot as plt
from GridWorld import standard_neg_grid, print_values, print_policy, action_space

gamma = 0.9

def play_game(g, P):
	start_states = list(g.actions.keys())
	start_index = np.random.choice(len(start_states))
	g.set_state(start_states[start_index])

	s = g.get_state()
	a = np.random.choice(action_space)

	state_action_reward = [(s,a,0)]
	seen_state = set()
	seen_state.add(g.get_state())
	itr = 0
	while True:
		r = g.move(a)
		itr += 1
		s = g.get_state()
		if s in seen_state:
			state_action_reward.append((s, None, -10./itr))
			break
		elif g.game_over():
			state_action_reward.append((s, None, r))
			break
		else:
			a = P[s]
			state_action_reward.append((s, a, r))
		seen_state.add(s)
	
	G = 0
	state_action_return = []
	first = True
	for s, a, r in reversed(state_action_reward):
		if first:
			first = False
		else:
			state_action_return.append((s, a, G))
		G = r + gamma*G
	state_action_return.reverse()
	return state_action_return

def max_dict(d):
	max_key = None
	max_val = float('-inf')
	for k,v in d.items():
		if v > max_val:
			max_val = v
			max_key = k
	return max_key, max_val

if __name__ == "__main__":
	g = standard_neg_grid(step_cost=-0.9)

# 	# Initilize Policy
	Policy = {}
	for s in g.actions.keys():
		Policy[s] = np.random.choice(action_space)

	# Initilize State value function Q and returns
	Q = {}
	returns ={}
	states = g.all_states()
	for s in states:
		if s in g.actions:
			Q[s] = {}
			for a in action_space:
				Q[s][a] = 0
				returns[(s, a)] = []
	
	deltas = []

	#Episodes
	for t in range(2000):
		print(t)
		delta = 0
		state_action_return = play_game(g, Policy)
		seen_state_action = set()
		for s, a, G in state_action_return:
			sa = (s,a)
			if sa not in seen_state_action:
				old_q = Q[s][a]
				returns[sa].append(G)
				Q[s][a] = np.mean(returns[sa])
				delta = max(delta, np.abs(old_q-Q[s][a]))
				seen_state_action.add(sa)
		deltas.append(delta)

		# update Policy
		for s in Policy.keys():
			Policy[s] = max_dict(Q[s])[0]

	plt.plot(deltas)
	plt.show()

	print("final Policy:")
	print_policy(Policy, g)

	# find V
	V = {}
	for s, Qs in Q.items():
	  V[s] = max_dict(Qs)[1]

	print("final values:")
	print_values(V, g)


