import numpy as np
from linear_model import *
class DQAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.model = LinearModel(state_size, action_size)

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return np.random.choice(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def train(self, state, action, reward, next_state, done):
		if done:
			target = reward
		else:
			target = reward + self.gamma*(np.amax(self.model.predict(next_state), axis=1))

		target_full = self.model.predict(state)
		target_full[0, action] = target

		self.model.sgd(state, target_full)

		if self.epsilon >= self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_params(name)

	def save(self, name):
		self.model.save_params(name) 


