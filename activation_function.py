import numpy as np

def sigmoid(x):
    # Sigmoid function : f(x) = 1/1+e^-x
    # Output range 0 to 1
    return 1/(1+np.exp(-x))

def tanh(x):
    # Tanh function
    # Output range -1 to 1
    numerator = 1 - np.exp(-2*x)
    denominator = 1 + np.exp(-2*x)
    return numerator/denominator

def ReLU(x):
    # Rectified Linear unit function
    # Retun 0 when x < zero
    if x<0:
        return 0
    else:
        return x

def leakyReLU(x,alpha=0.01):
    # leaky Rectified Linear unit function
    # Retun slop of negative value when x < zero
    if x<0:
        return (alpha*x)
    else:
        return x

def ELU(x,alpha=0.01):
    #The Exponential linear unit function
    # Like Relu it has a log curve
    if x<0:
        return ((alpha*(np.exp(x)-1)))
    else:
        return x

def swish(x,beta):
    #Swish By Google , better performance than ReLu
    # Swish is a non-monotonic function, which means it is neither always non-increasing nor non-decreasing.
    return 2*x*sigmoid(beta*x)

def softmax(x):
    # Softmax
    # generalization of the sigmoid function.
    # It is usually applied to the final layer of the network and while performing multi-class classification tasks.
    # It gives the probabilities of each class for being output and thus, the sum of softmax values will always equal 1.
    return np.exp(x) / np.exp(x).sum(axis=0)


def sigmoid_derivative(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

if __name__ == "__main__":
    for x in range(-100,100):
        print("X {} , VALUE {}".format(x,softmax(x)))
