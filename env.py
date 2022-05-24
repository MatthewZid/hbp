import math
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

class Env():
	#-------------------------
	# Constructor
	#-------------------------
	def __init__(self, speed=0.02, sigma1=0.01, sigma2=0.005, max_step=512):
		self.state    = None
		self.max_step = max_step
		self.n_step   = 1
		self.stop_crit = None
		self.sigma1   = sigma1
		self.sigma2   = sigma2
		self.speed    = speed
		self.p = None
	

	#-------------------------
	# Step
	#-------------------------
	def step(self, action):
		for i in range(self.p.shape[0]): self.p[i] += action[i] + self.sigma2*np.random.randn()
		self.n_step += 1

		for i in range(4):
			self.state[i, :] = self.state[i+1, :]
		self.state[4] = np.copy(self.p)

		reward = 0.0

		# if self.n_step >= 129 or abs(self.p[0]) >= 1 or abs(self.p[1]) >= 1:
		if self.p[-1] >= self.stop_crit:	# must determine stop criteria with object distance (here: if y-wrist coord >= stop_crit, stop)
			done = True
		else:
			done = False

		return np.copy(self.state.flatten()), done


	#-------------------------
	# Reset
	#-------------------------
	def reset(self, code, start=(0.0,0.0,0.0)):
		self.state = np.zeros((5, len(start)), dtype=np.float64)
		self.p = np.zeros((len(start),), dtype=np.float64)
		for i in range(len(start)): self.p[i] = start[i]
		self.n_step = 1

		for i in range(5):
			for j in range(len(start)):
				self.state[i,j] = self.p[j]
		
		# set trajectory stop criteria (y-wrist)
		if code[0] == 1: self.stop_crit = 168.6 # small
		elif code[1] == 1: self.stop_crit = 165.11 # medium
		elif code[2] == 1: self.stop_crit = 168.55 # large

		return np.copy(self.state.flatten())