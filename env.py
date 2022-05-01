import math
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

class Env():
	#-------------------------
	# Constructor
	#-------------------------
	def __init__(self, speed=0.02, sigma1=0.01, sigma2=0.005, max_step=512):
		self.state    = np.zeros((44,), dtype=np.float32)
		self.max_step = max_step
		self.n_step   = 1
		self.sigma1   = sigma1
		self.sigma2   = sigma2
		self.speed    = speed
	

	#-------------------------
	# Step
	#-------------------------
	def step(self, action):
		self.state += (action + self.sigma2*np.random.randn())
		self.n_step += 1

		reward = 0.0

		# if self.n_step >= 129 or abs(self.p[0]) >= 1 or abs(self.p[1]) >= 1:
		if self.n_step >= 46:
			done = True
		else:
			done = False

		return np.copy(self.state), done


	#-------------------------
	# Reset
	#-------------------------
	def reset(self, start):
		for i in range(44):
			self.state[i] = start[i]
		self.n_step = 1

		return np.copy(self.state)