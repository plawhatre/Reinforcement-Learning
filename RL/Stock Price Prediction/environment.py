import numpy as np
import itertools

class Environment:
	'''
	state = [number of stocks(3), price of per stock(3), uninvested amount(1) ] : 7 Dimensional 
	action = [sell, hold, buy] : 3 Dimensional
	'''
	def __init__(self, data, initial_investment=20000):
		#Data
		self.stock_price_history = data
		self.n_step, self.n_stock = self.stock_price_history.shape
		#Instance
		self.initial_investment = initial_investment
		self.current_step = None
		self.stock_owned = None
		self.stock_price = None
		self.cash_in_hand = None
		self.state_dim = self.n_stock*2 + 1

		self.action_space = np.arange(3**self.n_stock)
		self.action_list = list(map(list,itertools.product([0, 1, 2], repeat=self.n_stock)))

		self.reset()

	def reset(self):
		self.current_step = 0
		self.stock_owned = np.zeros(self.n_stock)
		self.stock_price = self.stock_price_history[self.current_step]
		self.cash_in_hand = self.initial_investment
		return self._get_obs()

	def step(self, action):
		assert action in self.action_space, 'Action not in action space'

		# portfolio value
		prev_val = self._get_val()

		self.current_step += 1
		self.stock_price = self.stock_price_history[self.current_step]

		self._trade(action)

		# portfolio value
		cur_val = self._get_val()

		#reward due to change in the portfolio value
		reward =cur_val - prev_val

		#if we run out of the data
		done = (self.current_step == self.n_step - 1)

		#store current value of the portfolio
		info = {'current value': cur_val}

		return  self._get_obs(), reward, done, info

	def _get_obs(self):
		obs = np.empty(self.state_dim)
		obs[:self.n_stock] = self.stock_owned
		obs[self.n_stock:2*self.n_stock] = self.stock_price
		obs[-1] = self.cash_in_hand
		return obs

	def _get_val(self):
		return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

	def _trade(self, action):
		action_vec =  self.action_list[action]

		sell_index = []
		buy_index = []
		for i,a in enumerate(action_vec):
			if a == 0:
				sell_index.append(i)
			if a == 2:
				buy_index.append(i)

		if sell_index:
			for i in sell_index:

				self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
				self.stock_owned[i] = 0

		if buy_index:
			can_buy = True
			while can_buy:
				for i in buy_index:
					if self.cash_in_hand > self.stock_price[i]:
						self.stock_owned[i] += 1 
						self.cash_in_hand -= self.stock_price[i]
					else:
						can_buy = False