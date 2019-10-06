import numpy as np

from activation_function import ActivationFunction

class Layer:
    ''' A.shape == (n_examples, output_size)
        Z.shape == (n_examples, output_size)
        W.shape == (input_size, output_size)
        b.shape == (output_size, )
        X.shape == (n_examples, input_size)
        obs.:
            input_size == prev_layer.output_size
            output_size == next_layer.input_size
    '''
    def __init__(self, output_size, activation_function, name=""):
        if activation_function != None:
            # obs.: the activation_function should only be None for the network's input layer
            assert(isinstance(activation_function, ActivationFunction)), "Invalid object type for activation_function"
        
        self.name = name
        
        self.input_size = None
        self.output_size = output_size
        
        # activation function
        self.g = activation_function # g_prime == activation_function.derivative
        
        # activation values
        self.A = None # self.A == self.g(self.Z)
        self.Z = None # prev_layer.A @ self.W + self.b
        
        # output value of the previous layer
        self.X = None # == prev_layer.A
        self.dX = None
        
        # parameters (weights and biases)
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
    
    def init(self, input_size, weight_initialization):
        ''' Sets the layer's input_size and initializes its weights and biases '''
        self.input_size = input_size
        if weight_initialization == 'xavier':
            stddev = np.sqrt(1 / self.input_size)
            self.W = stddev * np.random.randn(self.input_size, self.output_size)
            self.b = np.random.randn(self.output_size, )
        elif weight_initialization == 'xavier_avg':
            stddev = np.sqrt(2 / (self.input_size + self.output_size))
            self.W = stddev * np.random.randn(self.input_size, self.output_size)
            self.b = np.random.randn(self.output_size, )
        elif weight_initialization == 'rand_-1_to_1':
            self.W = 2 * np.random.randn(self.input_size, self.output_size) - 1
            self.b = 2 * np.random.randn(self.output_size, ) - 1
        else:
            raise ValueError(f"Invalid weight_initialization value: '{weight_initialization}'")
    
    @property
    def params_count(self):
        count = 0
        if self.W is not None:
            count += self.W.size
        if self.b is not None:
            count += self.b.size
        return count
    
    # receives the activation values of the previous layer (i.e. this layer's input)
    # returns the activation values of the current layer (i.e. next layer's input)
    def feedforward(self, X):
        ''' X.shape == (n_examples, self.input_size) '''
        assert(X.shape[1] == self.input_size)
        self.X = X
        # (n_examples, output_size) = (n_examples, input_size) @ (input_size, output_size) + (output_size, )
        self.Z = self.X @ self.W + self.b
        self.A = self.g(self.Z)
        return self.A
    
    # receives the derivative of the cost function w.r.t. the Z value of the current layer [dJ/dZ = dJ/dA . dA/dZ]
    # returns the derivative of the cost function w.r.t. the A value of the previous layer [dJ/dX = dJ/dZ . dZ/dX]
    # obs.: the A value of the previous layer is this layer's input value X
    def backprop(self, dZ):
        ''' dZ.shape == (n_examples, self.output_size)
        
            Note that only calling backprop doesn't actually update the layer parameters
        '''
        assert(dZ.shape[1] == self.output_size)
        # (input_size, output_size) = (input_size, n_examples)  @ (n_examples, output_size)
        # (output_size, )           = (n_examples, output_size).sum(axis=0)
        # (n_examples, input_size)  = (n_examples, output_size) @ (output_size, input_size), input_size==prev_layer.output_size
        self.dW = (self.X).T @ dZ # [dJ/dW = dJ/dZ . dZ/dX]
        self.db = dZ.sum(axis=0)  # [dJ/db = dJ/dZ . dZ/db]
        self.dX = dZ @ (self.W).T # [dJ/dX = dJ/dZ . dZ/dX]
        return self.dX
        # note that dJ/dX is dJ/dA for the previous layer (since this layer's input X is the previous layer's A)