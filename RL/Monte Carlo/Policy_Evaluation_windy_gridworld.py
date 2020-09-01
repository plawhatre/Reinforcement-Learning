import numpy as np
from GridWorld import standard_grid, print_values, print_policy, action_space

gamma = 0.9
delta = 10e-04

def random_action(a):
	p = np.random.random()
	if p < 0.5:
		return a
	else:
		tmp  = list(action_space)
		tmp.remove(a)
		return np.random.choice(tmp)

def play_game(g,P):

	#select a starting position in the grid world
	start_states = list(g.actions.keys())
	start_index = np.random.choice(len(start_states))
	g.set_state(start_states[start_index])
	# what is the current state in the grid
	s = g.get_state()

	#Rewards
	states_and_rewards = [(s,0)]
	while not g.game_over():
		a = P[s]
		a = random_action(a)
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

if __name__ == "__main__":
	g = standard_grid()

	print('rewards:')
	print_values(g.rewards, g)

	Policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'U',
    (2, 1): 'L',
    (2, 2): 'U',
    (2, 3): 'L',
  	}

	#Intilize V(s) and returns
	V = {}
	returns = {}
	states = g.all_states()
	for s in states:
		if s in g.actions:
			returns[s] = []
		else:
			V[s] = 0
	#repeat (Monte Carlo Loop)
	for t in range(100):
		states_and_returns = play_game(g, Policy)
		seen_state = set()
		for s, G in states_and_returns:
			if s not in seen_state:
				returns[s].append(G)
				V[s] = 	np.mean(returns[s])
				seen_state.add(s)
	
	print('Policy')
	print_policy(Policy, g)
	print('value')
	print_values(V, g)



