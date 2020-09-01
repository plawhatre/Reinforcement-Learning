import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
# Defined Modules
from utils import *
from environment import *
from linear_model import *
from agent import *

def play_one_episode(agent, env, is_train):
	state = env.reset()
	state = scaler.transform([state])
	done = False

	while not done:
		action = agent.act(state)
		next_state, reward, done, info = env.step(action)
		next_state = scaler.transform([next_state])
		if is_train == 'train':
			agent.train(state, action, reward, next_state, done)
		state = next_state

	return info['current value']

if __name__ == "__main__":
	model_folder = "Stock Price Prediction/trader_model"
	reward_folder = "Stock Price Prediction/trader_reward"
	num_episode = 2000
	batch_size = 32
	initial_investment = 20000

	args =str(input('Enter train or test:\t')).lower()

	maybe_make_dir(model_folder)
	maybe_make_dir(reward_folder)

	data = get_data()
	n_timesteps, n_stock = data.shape
	n_train = n_timesteps //2

	train_data = data[:n_train]
	test_data = data[n_train:]

	env = Environment(train_data, initial_investment)
	state_size = env.state_dim
	action_size = len(env.action_list)
	agent = DQAgent(state_size, action_size)
	scaler = get_scaler(env)

	portfolio_value = []

	if args == 'test':
		with open(f'{model_folder}/scaler.pkl','rb') as f:
			scaler = pickle.load(f)
		
		env = Environment(test_data, initial_investment)

		agent.epsilon = 0.01
	
		agent.load(f'{model_folder}/linear.npz')
			
	for e in range(num_episode):
		t0 = datetime.now()
		val = play_one_episode(agent, env, args)
		dt = datetime.now() - t0
		print(f"episode: {e + 1}/{num_episode}, episode end value: {val:.2f}, duration: {dt}")
		portfolio_value.append(val)

	if args == 'train':
		agent.save(f'{model_folder}/linear.npz')

		with open(f'{model_folder}/scaler.pkl','wb') as f:
			pickle.dump(scaler, f)

		plt.plot(agent.model.losses)
		plt.show()
	# save portfolio value for each episode
	np.save(f'{reward_folder}/{args}.npy', portfolio_value)
































