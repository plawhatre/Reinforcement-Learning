import numpy as np
import matplotlib.pyplot as plt

file = str(input('Enter train or test:\t')).lower()

try:
	try:
		a = np.load(f'trader_reward/{file}.npy')
	except:
		a = np.load(f'Stock Price Prediction/trader_reward/{file}.npy')

	print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

	plt.hist(a, bins=20)
	plt.title(file[0].upper()+file[1:]+' Mode')
	plt.show()
except:
	print('Input is not proper.')
