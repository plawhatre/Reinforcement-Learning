import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def get_data():
	try:
		df = pd.read_csv('data/aapl_msi_sbux.csv')
	except:
		df = pd.read_csv('Stock Price Prediction/data/aapl_msi_sbux.csv')
	return df.to_numpy()

def get_scaler(env):
	states = []
	for _ in range(env.n_step):
		action = np.random.choice(env.action_space)
		state, reward, done, info = env.step(action)
		states.append(state)
		if done:
			break
	scaler = StandardScaler()
	scaler.fit(states)
	return scaler

def maybe_make_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
