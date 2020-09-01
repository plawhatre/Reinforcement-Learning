import numpy as np
from GridWorld import standard_grid, action_space, print_values, print_policy

delta = 10e-4
gamma = 0.9
alpha = 0.1

def random_action(a, eps=0.1):
	p = np.random.random()

	if p< (1-eps):
		return a
	else:
		return np.random.choice(action_space)

def play_game(g, P):
	s = (2,0)
	g.set_state(s)
	state_reward = [(s,0)]
	while not g.game_over():
		a = random_action(P[s])
		r = g.move(a)
		s = g.get_state()
		state_reward.append((s,r))
	return state_reward

if __name__ == "__main__":
	g = standard_grid()
	Policy = {
	(2, 0): 'U',
	(1, 0): 'U',
	(0, 0): 'R',
	(0, 1): 'R',
	(0, 2): 'R',
	(1, 2): 'R',
	(2, 1): 'R',
	(2, 2): 'R',
	(2, 3): 'U',
	}
	#Initializ V
	V = {}
	for s in g.all_states():
		V[s] = 0
	#Iterations
	for itr in range(1000):
		state_reward = play_game(g, Policy)
		for t in range(len(state_reward)-1):
			s, _ = state_reward[t]
			s2, r = state_reward[t+1]
			V[s] = V[s] + alpha*(r + gamma*V[s2]-V[s])
	print("values")
	print_values(V, g)
	print("policy")
	print_policy(Policy, g) 

