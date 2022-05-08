import math
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

class Env():
	#-------------------------
	# Constructor
	#-------------------------
	def __init__(self, speed=0.02, sigma1=0.01, sigma2=0.005, max_step=512):
		self.state    = np.zeros((5,), dtype=np.float32)
		self.max_step = max_step
		self.n_step   = 1
		self.sigma1   = sigma1
		self.sigma2   = sigma2
		self.speed    = speed
		self.p = np.zeros((4,), dtype=np.float32)
	

	#-------------------------
	# Step
	#-------------------------
	def step(self, action):
		for i in range(4): self.p[i] += action[i] + self.sigma2*np.random.randn()
		aperture = np.sqrt((self.p[0] - self.p[2])**2 + (self.p[1] - self.p[3])**2)
		self.n_step += 1

		for i in range(4):
			self.state[i] = self.state[i+1]
		self.state[4] = aperture

		reward = 0.0

		# if self.n_step >= 129 or abs(self.p[0]) >= 1 or abs(self.p[1]) >= 1:
		if self.n_step >= 40:
			done = True
		else:
			done = False

		return np.copy(self.state), done


	#-------------------------
	# Reset
	#-------------------------
	def reset(self, start=(0.0,0.0,0.0,0.0)):
		for i in range(4): self.p[i] = start[i]
		aperture = np.sqrt((self.p[0] - self.p[2])**2 + (self.p[1] - self.p[3])**2)
		self.n_step = 1

		for i in range(5):
				self.state[i] = aperture

		return np.copy(self.state.flatten())