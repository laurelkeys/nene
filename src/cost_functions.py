import numpy as np

from activation_functions import ActivationFunction, SoftMax, Linear

class CostFunction:
    ''' A CostFunction is applied to Y (the target values) and Ypred (the predicted values) to get a scalar output
        Its derivative w.r.t. Ypred also expects Y and Ypred, but returns a tensor of shape (n_examples, last_layer_output_size)
        
        obs.: Ypred is the last layer's activation values: last_layer.A == last_layer.g(last.layer.Z), 
              i.e. last_layer is the output layer of the network
    '''
    def __call__(self, Y, Ypred):
        ''' Y.shape == Ypred.shape == (n_examples, last_layer_output_size) '''
        raise NotImplementedError # [J(Y, Ypred) == J(Y, A^L)]
    def derivative(self, Y, Ypred):
        ''' Y.shape == Ypred.shape == (n_examples, last_layer_output_size) '''
        raise NotImplementedError # [dJ/dYpred == dJ/dA^L]
    def deltaL(self, Y, Ypred, activation_function):
        ''' Y.shape == Ypred.shape == (n_examples, last_layer_output_size) 

            activation_function should be the last layer's g function, thus Ypred == activation_function(Z)
            and the value returned is the delta for the output layer of the network (delta^L == dJ/dZ^L)
        '''
        return self.derivative(Y, Ypred) * activation_function.derivative(A=Ypred) # [dJ/dZ^L == dJ/dYpred . dYpred/dZ^L]
        # obs.: Ypred == A^L == g(Z^L), thus dYpred/dZ^L == dA^L/dZ^L == g'(Z^L)
        #       [dJ/dZ^L == dJ/dYpred . dYpred/dZ^L == dJ/dA^L . dA^L/dZ^L]

class CrossEntropy(CostFunction):
    def __call__(self, Y, Ypred, eps=1e-9):
        return np.mean( -(Y * np.log(Ypred+eps)).sum(axis=1) )
    def derivative(self, Y, Ypred, eps=1e-9):
        m = Ypred.shape[0]
        return - (Y / (Ypred+eps)) / m
    def deltaL(self, Y, Ypred, activation_function):
        if isinstance(activation_function, SoftMax):
            m = Ypred.shape[0]
            # numerically stable
            return (Ypred - Y) / m # (SoftMax(Z) - Y) / m
        else:
            return super().deltaL(Y, Ypred, activation_function)

class SoftmaxCrossEntropy(CostFunction):
    def __call__(self, Y, Ypred):
        exp = np.exp(Ypred - Ypred.max(axis=1, keepdims=True))
        Softmax = exp / np.sum(exp, axis=1, keepdims=True)
        return np.mean( -(Y * np.log(Softmax)).sum(axis=1) )
    def derivative(self, Y, Ypred):
        exp = np.exp(Ypred - Ypred.max(axis=1, keepdims=True))
        Softmax = exp / np.sum(exp, axis=1, keepdims=True)
        m = Softmax.shape[0]
        return (Softmax - Y) / m
    def deltaL(self, Y, Ypred, activation_function):
        if isinstance(activation_function, Linear):
            # Linear.derivative(Z) is a matrix of ones, so 
            # calling it doesn't change the returned value
            return self.derivative(Y, Ypred) # (SoftMax(Z) - Y) / m
        else:
            return super().deltaL(Y, Ypred, activation_function)