import numpy as np
from GridWorld import standard_grid, standard_neg_grid, action_space, print_policy, print_values
import matplotlib.pyplot as plt

gamma = 0.9
Alpha = 0.001

def play_game(g,P):

	#select a starting position in the grid world
	start_state = list(g.actions.keys())
	start_index = np.random.choice(len(start_state))
	g.set_state(start_state[start_index])
	# what is the current state in the grid
	s = g.get_state()

	#Rewards
	states_and_rewards = [(s,0)]
	while not g.game_over():
		a = P[s]
		r = g.move(a)
		s = g.get_state()
		states_and_rewards.append((s,r))

	#Return
	G = 0
	states_and_returns = []
	first = True
	for s,r in reversed(states_and_rewards):
		if first:
			first  = False
		else:
			states_and_returns.append((s,G))
		G = r + gamma*G
	states_and_returns.reverse()
	return states_and_returns

def random_action(a):
	p = np.random.random()
	if p < 0.5:
		return a
	else:
		tmp  = list(action_space)
		tmp.remove(a)
		return np.random.choice(tmp)

if __name__ == "__main__":
	g = standard_grid()

	Policy = {
	(2, 0): 'U',
	(1, 0): 'U',
	(0, 0): 'R',
	(0, 1): 'R',
	(0, 2): 'R',
	(1, 2): 'U',
	(2, 1): 'L',
	(2, 2): 'U',
	(2, 3): 'L'
	}
	# phi(s) = [row, col, row*col, 1]
	#Theta Intilization
	theta = np.random.randn(4)
	def s2x(s):
		return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])

	deltas = []

	t = 1.0

	for itr in range(2000):
		if itr%100 == 0:
			t += 0.01
		alpha = Alpha/t
		delta = 0
		state_and_return = play_game(g, Policy)
		seen_state = set()
		for s, G in state_and_return:
			if s not in seen_state:
				old_theta = theta.copy()
				x = s2x(s)
				V_hat = theta.dot(x)
				theta += alpha*(G-V_hat)*x
				delta = max(delta, np.abs(theta - old_theta).sum())
				seen_state.add(s)
		deltas.append(delta)

	plt.plot(deltas)
	plt.show()

	#Calculate V
	V = {}
	for s in g.all_states():
		if s in g.actions.keys():
			V[s] = np.dot(theta, s2x(s))
		else:
			V[s] = 0

	# print("Values")
	print_values(V, g) 
	print("Policy")
	print_policy(Policy, g)



