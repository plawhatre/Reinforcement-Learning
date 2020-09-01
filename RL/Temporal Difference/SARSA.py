import numpy as np
import matplotlib.pyplot as plt
from GridWorld import standard_neg_grid, action_space, print_values, print_policy

gamma = 0.9
Alpha = 0.1

def max_dict(d):
	max_key = None
	max_val = float('-inf')
	for k,v in d.items():
		if v > max_val:
			max_val = v
			max_key = k
	return max_key, max_val

def random_action(a, eps=0.1):
	p = np.random.random()

	if p< (1-eps):
		return a
	else:
		return np.random.choice(action_space)

#No play_game function since TD is completly online
if __name__ == "__main__":
	g = standard_neg_grid()

	# Initialize Q and adaptive learning rate
	Q = {}
	update_count_sa = {}
	for s in g.all_states():
		Q[s] = {}
		update_count_sa[s] = {}
		for a in action_space:
			Q[s][a] = 0
			update_count_sa[s][a] = 1.0

	# Repeat till convergence
	t = 1
	deltas = []
	for itr in range(10000):
		if itr%100 == 0:
			t += 1e-03
		if itr%1000 == 0:
			print(itr)

		s = (2, 0)
		g.set_state(s)
		a = max_dict(Q[s])[0]
		a = random_action(a,eps= 0.5/t)
		delta = 0
		# play game
		while not g.game_over():
			r = g.move(a)
			s2 = g.get_state()

			a2 = max_dict(Q[s2])[0]
			a2 = random_action(a2,eps=0.5/t)

			alpha = Alpha / update_count_sa[s][a]
			update_count_sa[s][a] += 0.005

			old_qsa = Q[s][a]
			Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s2][a2]-Q[s][a])

			delta = max(delta, np.abs(old_qsa-Q[s][a]))

			s = s2
			a = a2

		deltas.append(delta)

	plt.plot(deltas)
	plt.show()

	#Find Policy and V function
	Policy = {}
	V = {}
	for s in g.actions.keys():
		a, max_q = max_dict(Q[s])
		Policy[s] = a
		V[s] = max_q

	print("Values")
	print_values(V, g)
	print("Policy")
	print_policy(Policy, g)
