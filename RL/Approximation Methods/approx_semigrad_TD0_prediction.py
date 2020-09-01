import numpy as np
from GridWorld import standard_grid, action_space, print_values, print_policy
import matplotlib.pyplot as plt

delta = 10e-4
gamma = 0.9
Alpha = 0.1

class Model:
	def __init__(self):
		self.theta = np.random.randn(4)/2

	#policy can be improved if better features are engineered to approxiamte tha value function
	def s2x(self, s):
		return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])

	def predict(self, s):
		return np.dot(self.theta, self.s2x(s))

	def grad(self, s):
		return self.s2x(s)

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

	model = Model()
	deltas = []

	k = 1.0
	for itr in range(30000):
		if itr%100 == 0:
			k += 0.01
			print(itr)
		alpha = Alpha/k
		delta = 0
		state_reward = play_game(g, Policy)
		for t in range(len(state_reward)-1):
			s, _ = state_reward[t]
			s2, r = state_reward[t+1]

			old_theta = model.theta.copy()
			if g.is_terminal(s2):
				target = r
			else:
				target = r + gamma * model.predict(s2)

			model.theta += alpha*(target - model.predict(s))*model.grad(s)
			delta = max(delta, np.abs(model.theta - old_theta).sum())
		deltas.append(delta)
	
	plt.plot(deltas)
	plt.show()

	V = {}
	for s in g.all_states():
		if s in g.actions.keys():
			V[s] = np.dot(model.theta, model.s2x(s))
		else:
			V[s] = 0
	
	print("values")
	print_values(V, g)
	print("policy")
	print_policy(Policy, g) 

