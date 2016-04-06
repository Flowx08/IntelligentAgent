#Import modules
import numpy as np
import math
import random
import copy
import NeuralNet as nn

#Experience informations
class Experience:
	def __init__(self):
		self.state = []
		self.next_state = []
		self.actionid = 0
		self.reward = 0.0

#Intelligent Agent 
class IntelligentAgent:
	
	def __init__(self, short_term_mem_size, learningrate, curiosity, lt_exploitation):
		#Create neural network layers
		self.netL1 = nn.FullyConnectedNL(49, 25, "RELU", "DRELU", 0.0, 0.1)
		self.netL2 = nn.FullyConnectedNL(25, 4, "LINEAR", "DLINEAR", 0.0, 0.1)
		self.short_term_mem = []
		self.short_term_mem_size = short_term_mem_size
		self.short_term_mem_pos = 0
		self.current_experience = Experience()
		self.curiosity = curiosity
		self.lt_exploitation = lt_exploitation
		self.learningrate = learningrate
		self.error = 0.0

	def fire(self, data):
		self.netL1.fire(data)
		self.netL2.fire(self.netL1.outputs)
		return self.netL2.outputs
	
	def train(self, inputs, outputs):
		self.fire(inputs)
		self.netL2.calculate_output_deltas(outputs)
		self.netL2.backpropagate_deltas(self.netL1)
		self.netL2.update_weights(self.netL1.outputs, self.learningrate)
		self.netL1.update_weights(inputs, self.learningrate)
		self.error = np.absolute(np.subtract(self.netL2.outputs, outputs)).sum()

	def step(self, state):
		if len(state) == 0: return
		self.current_experience.state = copy.copy(state)
		if (np.random.uniform() < self.curiosity):
			self.current_experience.actionid = np.random.randint(2)
		else:
			self.current_experience.actionid = np.argmax(self.fire(state))
		return self.current_experience.actionid
		
	def learn(self, next_state, reward):
		if len(self.current_experience.state) == 0: return
		self.current_experience.next_state = copy.copy(next_state)
		self.current_experience.reward = copy.copy(reward)

		if (len(self.short_term_mem) < self.short_term_mem_size):
			self.short_term_mem.append(copy.copy(self.current_experience))
		else:
			self.short_term_mem[self.short_term_mem_pos] = copy.copy(self.current_experience)
			self.short_term_mem_pos += 1
			if (self.short_term_mem_pos >= self.short_term_mem_size): self.short_term_mem_pos = 0
		
		for i in xrange(3):
			r = np.random.randint(len(self.short_term_mem))
			experience_reward = 0.0
			if len(self.short_term_mem[r].next_state) == 0:
				experience_reward = self.short_term_mem[r].reward
			else:
				nextmaxreward = np.amax(self.fire(next_state))
				experience_reward = self.short_term_mem[r].reward + self.lt_exploitation * nextmaxreward
			actions = copy.copy(self.fire(self.short_term_mem[r].state))
			actions[self.short_term_mem[r].actionid] = experience_reward
			self.train(self.short_term_mem[r].state, actions)
	
	def save(self):
		self.netL1.save("netL1")
		self.netL2.save("netL2")
	
	def load(self):
		try:
			self.netL1.load("netL1")
			self.netL2.load("netL2")
		except:
			print("ERROR: Can't load neural network from file...")


