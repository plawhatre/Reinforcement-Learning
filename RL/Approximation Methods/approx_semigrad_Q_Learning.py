'''
This method does not have a good convergence
'''
import numpy as np
import matplotlib.pyplot as plt
from GridWorld import standard_neg_grid, action_space, print_values, print_policy

delta = 1e-05
gamma = 0.9
Alpha = 0.001
idx = 0

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

class Model:
	def __init__(self):
		self.theta = np.random.randn(25) / np.sqrt(25)

	def sa2x(self, s, a):
		return np.array([
			s[0] - 1              if a == 'U' else 0,
			s[1] - 1.5            if a == 'U' else 0,
			(s[0]*s[1] - 3)/3     if a == 'U' else 0,
			(s[0]*s[0] - 2)/2     if a == 'U' else 0,
			(s[1]*s[1] - 4.5)/4.5 if a == 'U' else 0,
			1                     if a == 'U' else 0,
			s[0] - 1              if a == 'D' else 0,
			s[1] - 1.5            if a == 'D' else 0,
			(s[0]*s[1] - 3)/3     if a == 'D' else 0,
			(s[0]*s[0] - 2)/2     if a == 'D' else 0,
			(s[1]*s[1] - 4.5)/4.5 if a == 'D' else 0,
			1                     if a == 'D' else 0,
			s[0] - 1              if a == 'L' else 0,
			s[1] - 1.5            if a == 'L' else 0,
			(s[0]*s[1] - 3)/3     if a == 'L' else 0,
			(s[0]*s[0] - 2)/2     if a == 'L' else 0,
			(s[1]*s[1] - 4.5)/4.5 if a == 'L' else 0,
			1                     if a == 'L' else 0,
			s[0] - 1              if a == 'R' else 0,
			s[1] - 1.5            if a == 'R' else 0,
			(s[0]*s[1] - 3)/3     if a == 'R' else 0,
			(s[0]*s[0] - 2)/2     if a == 'R' else 0,
			(s[1]*s[1] - 4.5)/4.5 if a == 'R' else 0,
			1                     if a == 'R' else 0,
			1 ])

	def predict(self, s, a):
		return np.dot(self.theta, self.sa2x(s, a))

	def grad(self, s, a):
		return self.sa2x(s, a)

def getQs(model, s):
	Qs = {}
	for a in action_space:
		q_sa = model.predict(s, a)
		Qs[a] = q_sa
	return Qs

if __name__ == "__main__":
	g = standard_neg_grid()

	model = Model()
	# Repeat till convergence
	t = 1.0
	t2 = 1.0
	deltas = []
	for itr in range(20000):
		if itr%100 == 0:
			t += 0.01
			t2 += 0.01
		if itr%1000 == 0:
			print(itr)

		alpha = Alpha / t2
		s = (2, 0)
		g.set_state(s)
		Qs = getQs(model, s)
		a = max_dict(Qs)[0]
		a = random_action(a,eps= 0.5/t)
		delta = 0
		# play game
		while not g.game_over():
			r = g.move(a)
			s2 = g.get_state()
			
			old_theta = model.theta.copy()

			if g.is_terminal(s2):
				model.theta += alpha*(r - model.predict(s, a))*model.grad(s, a) 
			else:
				Qs2 = getQs(model, s2)
				a2, MaxQs2a2 = max_dict(Qs2)
				a2 = random_action(a2,eps = 0.5/t)
				model.theta += alpha*(r + gamma*MaxQs2a2 - model.predict(s, a))*model.grad(s, a) 
				s = s2
				a = a2

			delta = max(delta, np.abs(old_theta-model.theta).sum())

		deltas.append(delta)

	plt.plot(deltas)
	plt.show()

	#Find Policy and V function
	Policy = {}
	V = {}
	Q = {}
	for s in g.actions.keys():
		Q[s] = getQs(model, s)
		a, max_q = max_dict(Q[s])
		Policy[s] = a
		V[s] = max_q

	print("Values")
	print_values(V, g)
	print("Policy")
	print_policy(Policy, g)


