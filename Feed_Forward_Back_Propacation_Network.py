import numpy as np
import activation_function as af
import cost_function as cf
import matplotlib.pyplot as plt



def two_layer_backword_propagation(y_hat, z1, a1, z2):
	# Backword Propagation
	delta2 = np.multiply(-(y-y_hat),af.sigmoid_derivative(z2))
	dJ_dWhy = np.dot(a1.T, delta2)
	delta1 = np.dot(delta2,Why.T)*af.sigmoid_derivative(z1)
	dJ_dWxh = np.dot(x.T, delta1) 

	return dJ_dWxh, dJ_dWhy

def two_layer_forward_propagation(input_X,Wxh,Why):
	# Forward Propagation
	# two layers = one hidden layers + output layers (not input layer)
	
	

	#print("input nodes = {}, Output nodes = {}, Hidden nodes = {}, input = {}".format(
	#	num_input_nodes,num_output_nodes,num_hidden_nodes,input_X))

	# multiply inputs with weight and add bias 

	z1 = np.dot(input_X,Wxh)+bh

	# Apply activation function from z1

	a1 = af.sigmoid(z1)

	# multiply hidden layer output with hideent to output layer weight and add bias (by) 

	z2 = np.dot(a1,Why)+by

	# Apply activation function from z2 , y_hat is the predicted output

	y_hat = af.sigmoid(z2)

	return z1,a1,z2,y_hat


num_iterations = 100000

# alpha = learning rate
alpha = 0.01

# x = input , y = output
x = np.array([ [0, 1], [1, 0], [1, 1],[0, 0] ])
y = np.array([ [1], [1], [0], [0]])
num_input_nodes = 2
num_hidden_nodes = 5
num_output_nodes = 1

#Initializing the input to hidden layer weights ( W = weight , x = inputlayer , y = hidden layer)
Wxh = np.random.randn(num_input_nodes,num_hidden_nodes)

"""Wxh = array([[ 0.82632263,  0.04863786, -0.99543347,  1.09977506, -0.01788192],
	   [ 1.8583379 ,  0.72674049, -0.52315189,  0.08762322, -0.61746624]])
	"""

#Initializing the hidden to output layer weights ( W = weight , h = hiddenlayer , y = output layer)
Why = np.random.randn(num_hidden_nodes,num_output_nodes)

"""Why = array([[ 1.789417  ],
	   [-1.37029289],
	   [-0.23101313],
	   [-0.08765128],
	   [-0.93240217]])
	"""

# Initializing the bias for input to hidden layer
bh = np.zeros((1,num_hidden_nodes))
	
""" bh = array([[0., 0., 0., 0., 0.]]) """

# Initializing the bias for input to hidden layer
by = np.zeros((1,num_output_nodes))

""" by = array([[0.]]) """

cost = []
for i in range(num_iterations):
	#perform forward propagation and predict output
	z1,a1,z2,y_hat = two_layer_forward_propagation(x,Wxh,Why)
	#perform backward propagation and calculate gradients
	dJ_dWxh, dJ_dWhy = two_layer_backword_propagation(y_hat, z1, a1, z2)

	#update the weights
	Wxh = Wxh -alpha * dJ_dWxh
	Why = Why -alpha * dJ_dWhy
	
	#compute cost
	c = cf.mean_squared_error(y, y_hat)
	
	#store the cost
	print(c)
	cost.append(c)

plt.grid()
plt.plot(range(num_iterations),cost)

plt.title('Cost Function')
plt.xlabel('Training Iterations')
plt.ylabel('Cost')
plt.show()

# Testing the network : giving input [1,1] using learned weight Wxh,Why
z1,a1,z2,y_hat = two_layer_forward_propagation([1,1],Wxh,Why)
print(y_hat)