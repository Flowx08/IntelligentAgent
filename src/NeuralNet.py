################################
# Modules
################################
import random
import math
import numpy as np


################################
# Activation functions
################################
def activation_relu(val):
	return max(0, val)

def activation_relu_deriv(val):
	if val > 0: return 1
	else: return 0

def activation_linear(val):
	return val

def activation_linear_deriv(val):
	return 1

def activation_tanh(val):
	return math.tanh(val)

def activation_tanh_deriv(val):
	return 1.0 - math.tanh(val)**2

def activation_tostring(activation):
	if (activation == activation_relu): return "RELU"
	elif (activation == activation_linear): return "LINEAR"
	elif (activation == activation_tanh): return "TANH"
	else: return "UNKN"

def string_toactivation(string):
	if (string == "RELU"): return activation_relu
	elif (string == "LINEAR"): return activation_linear
	elif (string == "TANH"): return activation_tanh
	else:
		print("Unknown activation '{}'".format(string))
		return 0x0

def derivative_tostring(derivative):
	if (derivative == activation_relu_deriv): return "DRELU"
	elif (derivative == activation_linear_deriv): return "DLINEAR"
	elif (derivative == activation_tanh_deriv): return "DTANH"
	else: return "DUNKN"

def string_toderivative(string):
	if (string == "DRELU"): return activation_relu_deriv
	elif (string == "DLINEAR"): return activation_linear_deriv
	elif (string == "DTANH"): return activation_tanh_deriv
	else:
		print("Unknown activation '{}'".format(string))
		return 0x0
	

################################
# Fully connected neural layer
################################
class FullyConnectedNL:
	
	def __init__(self, inputs_size, nodes_size, activation="LINEAR", derivative="DLINEAR", init_mean=0.0, init_dev=1.0):
		if nodes_size <= 0 or inputs_size <= 0: 
			print("Error, invalid FullyConnected_layer init parameters")
		self.nodes = nodes_size
		self.weights = np.random.normal(loc=init_mean, scale=init_dev, size=(nodes_size, inputs_size))
		self.bias = np.random.normal(loc=init_mean, scale=init_dev, size=(nodes_size))
		self.outputs = np.zeros(nodes_size)
		self.deltas = np.zeros(nodes_size)
		self.activation = string_toactivation(activation)
		self.derivative = string_toderivative(derivative)

	def fire(self, inputs):
		for i in range(self.nodes):
			out = np.dot(self.weights[i], inputs)
			out += self.bias[i]
			self.outputs[i] = self.activation(out)	
			
	def backpropagate_deltas(self, back_layer):
		for i in range(back_layer.nodes):
			error = math.fsum([(self.deltas[l] * self.weights[l][i]) for l in range(self.nodes)])
			back_layer.deltas[i] = self.derivative(back_layer.outputs[i]) * error

	def calculate_output_deltas(self, outputs):	
		for i in range(self.nodes):
			self.deltas[i] = self.derivative(self.outputs[i]) * ((outputs[i] - self.outputs[i]))

	def update_weights(self, inputs, learning_rate):
		for i in range(self.nodes):
			for k in range(len(inputs)):
				self.weights[i][k] += inputs[k] * learning_rate * self.deltas[i]
			self.bias[i] += learning_rate * self.deltas[i];
	
	def clear(self):
		self.outputs = np.zeros(len(self.outputs))
	
	def save(self, filepath):
		f = file(filepath, "wb")
		np.save(f, self.weights)
		np.save(f, self.bias)
		f.write(activation_tostring(self.activation) + "\n")
		f.write(derivative_tostring(self.derivative) + "\n")
		f.close()
	
	def load(self, filepath):
		f = file(filepath, "rb")
		self.weights = np.load(f)
		self.bias = np.load(f)
		self.activation = string_toactivation(f.readline()[:-1])
		self.derivative = string_toderivative(f.readline()[:-1])
		f.close()

################################
# Recurrent neural layer
################################
class RecurrentNL:
	
	def __init__(self, inputs_size, nodes_size, activation="LINEAR", derivative="DLINEAR", init_mean=0.0, init_dev=1.0):
		if nodes_size <= 0 or inputs_size <= 0: 
			print("Error, invalid Recurrent_layer init parameters")
		self.nodes = nodes_size
		self.weights = np.random.normal(loc=init_mean, scale=init_dev, size=(nodes_size, inputs_size + nodes_size))
		self.bias = np.random.normal(loc=init_mean, scale=init_dev, size=nodes_size)
		self.outputs = np.zeros(nodes_size)
		self.deltas = np.zeros(nodes_size)
		self.inputs = np.zeros(inputs_size + nodes_size)
		self.activation = string_toactivation(activation)
		self.derivative = string_toderivative(derivative)
	
	def fire(self, inputs):
		self.inputs = np.concatenate((inputs, self.outputs), axis=0)
		for i in range(self.nodes):
			out = np.dot(self.weights[i], self.inputs)
			out += self.bias[i]
			self.outputs[i] = self.activation(out)
	
	def backpropagate_deltas(self, back_layer):
		for i in range(back_layer.nodes):
			error = math.fsum([(self.deltas[l] * self.weights[l][i]) for l in range(self.nodes)])
			back_layer.deltas[i] = self.derivative(self.outputs[i]) * error

	def calculate_output_deltas(self, outputs):	
		for i in range(self.nodes):
			self.deltas[i] = self.derivative(self.outputs[i]) * (outputs[i] - self.outputs[i])

	def update_weights(self, inputs, learning_rate):
		for i in range(self.nodes):
			for k in range(len(self.inputs)):
				self.weights[i][k] += self.inputs[k] * learning_rate * self.deltas[i]
			self.bias += learning_rate * self.deltas[i]	
	
	def clear(self):
		self.outputs = np.zeros(len(self.outputs))
	
	def save(self, filepath):
		f = file(filepath, "wb")
		np.save(f, self.weights)
		np.save(f, self.bias)
		f.write(activation_tostring(self.activation) + "\n")
		f.write(derivative_tostring(self.derivative) + "\n")
		f.close()
	
	def load(self, filepath):
		f = file(filepath, "rb")
		self.weights = np.load(f)
		self.bias = np.load(f)
		self.activation = string_toactivation(f.readline())
		self.derivative = string_toderivative(f.readline())
		f.close()
